"""
The end-to-end XAI framework: one object that trains the ANN and answers the
three questions every stakeholder asks.

    GLOBAL  - "How does the model think, in general?"   (risk & strategy teams)
    LOCAL   - "Why THIS customer?"                       (relationship managers)
    TRUST   - "Why should I believe these explanations?" (model-risk, regulators)

FEYNMAN NOTE
------------
Think of the ANN as a sealed machine with a slot for a customer card and a dial
that shows "risk of leaving". We never open the casing by force. Instead we:
  - feed it doctored cards and watch the dial (SHAP, LIME, PDP, permutation),
  - read the dial's sensitivity with a stethoscope (gradients),
  - hire an apprentice to imitate it with a flow-chart (surrogate tree),
  - ask "what card would make the dial drop?" (counterfactuals),
  - and finally check our own detective work (trust module).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import FEATURES, Dataset, load_data, prepare
from .explainers.base import LocalExplanation
from .explainers.counterfactual import Counterfactual, CounterfactualExplainer
from .explainers.global_methods import (DependenceCurve, GlobalSurrogate, PartialDependence,
                                        PermutationImportance, SurrogateResult)
from .explainers.gradients import IntegratedGradientsExplainer
from .explainers.lime import LimeExplainer
from .explainers.shapley import ShapleyExplainer
from .model import AttritionANN, evaluate, train_ann
from .narrator import narrate
from .trust import agreement_report, faithfulness_report, ground_truth_recovery


@dataclass
class CustomerCase:
    customer_id: object
    label: str                                  # why this customer was picked
    actual_left: bool
    explanations: dict[str, LocalExplanation]
    counterfactual: Counterfactual | None
    story: str


@dataclass
class XAIResults:
    source: str
    metrics: dict
    base_rate: float
    permutation: pd.DataFrame
    shap_matrix: pd.DataFrame
    shap_sample: pd.DataFrame
    dependence: list[DependenceCurve]
    surrogate: SurrogateResult
    cases: list[CustomerCase]
    faithfulness: dict
    agreement: pd.DataFrame
    ground_truth: dict | None
    extras: dict = field(default_factory=dict)


class AttritionXAI:
    def __init__(self, csv_path: str | None = None, seed: int = 7, background_size: int = 200,
                 global_sample: int = 150, fast: bool = False):
        self.seed = seed
        self.fast = fast
        df, source = load_data(csv_path, seed=seed)
        self.simulated = source.startswith("simulated")
        self.data: Dataset = prepare(df, source, seed=seed)
        self.background = self.data.X_train.sample(background_size, random_state=seed).reset_index(drop=True)
        self.global_sample = global_sample if not fast else 40
        self.model: AttritionANN | None = None

    # ------------------------------------------------------------------ steps
    def fit(self) -> "AttritionXAI":
        self.model = train_ann(self.data, seed=self.seed)
        f = self.model.predict_proba
        n_perm = 32 if self.fast else 128
        self.shap = ShapleyExplainer(f, self.background, n_permutations=n_perm, seed=self.seed)
        self.shap_global = ShapleyExplainer(f, self.background, n_permutations=n_perm // 2, seed=self.seed)
        self.lime = LimeExplainer(f, self.background, n_samples=1000 if self.fast else 3000, seed=self.seed)
        self.ig = IntegratedGradientsExplainer(self.model, self.background)
        self.cf = CounterfactualExplainer(self.model, self.data.X_train)
        return self

    def explain_customer(self, x: pd.Series, customer_id=None, label: str = "") -> CustomerCase:
        x = x[FEATURES]
        ex = {"shapley": self.shap.explain(x), "lime": self.lime.explain(x), "ig": self.ig.explain(x)}
        cf = self.cf.explain(x) if ex["shapley"].prediction >= self.model.threshold else None
        typical = self.data.X_train[FEATURES].median(numeric_only=True)
        story = narrate(customer_id, ex["shapley"], ex["lime"], ex["ig"], cf, typical)
        return CustomerCase(customer_id, label, False, ex, cf, story)

    def _pick_cases(self, p: np.ndarray) -> list[tuple[int, str]]:
        y = self.data.y_test
        picks = [
            (int(np.argmax(np.where(y == 1, p, -1))), "Highest-risk customer who actually left"),
            (int(np.argmin(np.abs(p - 0.5))), "Borderline case (model on the fence)"),
            (int(np.argmin(np.where(y == 0, p, 2))), "Most confidently retained customer"),
            (int(np.argmax(np.where(y == 0, p, -1))), "False alarm: predicted to leave, but stayed"),
        ]
        return picks

    def run(self) -> XAIResults:
        if self.model is None:
            self.fit()
        d, f = self.data, self.model.predict_proba
        metrics = evaluate(self.model, d)
        p_test = f(d.X_test)

        # ---- GLOBAL -------------------------------------------------------
        perm = PermutationImportance(f, n_repeats=3 if self.fast else 5, seed=self.seed).compute(d.X_test, d.y_test)
        sample = d.X_test.sample(self.global_sample, random_state=self.seed)
        S = self.shap_global.explain_many(sample)
        top_feats = list(S.abs().mean().sort_values(ascending=False).index[:6])
        pdp = PartialDependence(f)
        dep_sample = d.X_test.sample(300, random_state=self.seed)
        curves = [pdp.compute(dep_sample, feat) for feat in top_feats]
        surrogate = GlobalSurrogate(self.model, seed=self.seed).fit(d.X_train, d.X_test)

        # ---- LOCAL --------------------------------------------------------
        cases = []
        for idx, label in self._pick_cases(p_test):
            case = self.explain_customer(d.X_test.iloc[idx], d.ids_test[idx], label)
            case.actual_left = bool(d.y_test[idx])
            cases.append(case)

        # ---- TRUST --------------------------------------------------------
        n_trust = 8 if self.fast else 25
        risky = d.X_test.iloc[np.argsort(-p_test)[:n_trust]]
        risky_ex = [self.shap.explain(risky.iloc[i]) for i in range(len(risky))]
        faith = faithfulness_report(f, risky_ex, self.background, n_random=3 if self.fast else 5,
                                    seed=self.seed)
        mixed = d.X_test.sample(n_trust, random_state=self.seed + 1)
        per_customer = []
        for i in range(len(mixed)):
            x = mixed.iloc[i]
            per_customer.append({"shapley": self.shap_global.explain(x),
                                 "lime": self.lime.explain(x), "ig": self.ig.explain(x)})
        agreement = agreement_report(per_customer, k=5)
        gt = ground_truth_recovery(S.abs().mean()) if self.simulated else None

        return XAIResults(d.source, metrics, d.base_rate, perm, S, sample, curves, surrogate,
                          cases, faith, agreement, gt,
                          extras={"lime_r2": float(np.mean([c["lime"].diagnostics["local_r2"]
                                                             for c in per_customer])),
                                  "ig_completeness_gap": float(np.max([abs(c["ig"].diagnostics["completeness_gap"])
                                                                       for c in per_customer])),
                                  "shap_additivity_gap": float(np.max([abs(c["shapley"].diagnostics["additivity_gap"])
                                                                       for c in per_customer]))})
