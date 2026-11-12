"""
Explaining the explanations: should we believe them?

FEYNMAN NOTE
------------
Simple idea:  An explanation is itself a claim, and claims need evidence. We run
              three checks a sceptical risk officer would ask for:

  1. Faithfulness ("deletion test"): if SHAP says facts A, B, C drive this
     customer's risk, then erasing A, B, C should collapse the risk faster than
     erasing random facts. If it doesn't, the explanation is decoration.
  2. Agreement: three different methods (outside poking: SHAP, LIME; inside
     looking: Integrated Gradients) should broadly name the same top drivers.
     Witnesses who agree are more credible.
  3. Ground truth: on simulated data we *planted* the real drivers. A good
     framework must rediscover them and rank demographics (planted as noise) low.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import CATEGORICAL_FEATURES, SIMULATED_TRUE_DRIVERS

DEMOGRAPHICS = ["Customer_Age", "Dependent_count", *CATEGORICAL_FEATURES]
from .explainers.base import LocalExplanation, rank_agreement


def deletion_curve(predict_fn, explanation: LocalExplanation, background: pd.DataFrame,
                   order: list[str] | None = None, n_bg: int = 64, seed: int = 0) -> np.ndarray:
    """P(leave) after replacing the first k features of ``order`` with background values.

    Default order = features ranked by how much they pushed risk UP (most first).
    """
    rng = np.random.default_rng(seed)
    x = explanation.customer
    if order is None:
        order = list(explanation.contributions.sort_values(ascending=False).index)
    bg = background.iloc[rng.integers(0, len(background), n_bg)].reset_index(drop=True)
    curve = []
    for k in range(len(order) + 1):
        rows = pd.DataFrame([x] * n_bg).reset_index(drop=True)
        for f in order[:k]:
            rows[f] = bg[f].to_numpy()
        curve.append(predict_fn(rows.astype(background.dtypes.to_dict())).mean())
    return np.array(curve)


def faithfulness_report(predict_fn, explanations: list[LocalExplanation], background: pd.DataFrame,
                        n_random: int = 5, seed: int = 0) -> dict:
    """Area under deletion curve: SHAP order vs random orders (lower = more faithful)."""
    rng = np.random.default_rng(seed)
    shap_auc, rand_auc, curves, rand_curves = [], [], [], []
    for e in explanations:
        c = deletion_curve(predict_fn, e, background, seed=seed)
        curves.append(c)
        shap_auc.append(np.trapezoid(c) / len(c))
        feats = list(e.contributions.index)
        r = [deletion_curve(predict_fn, e, background, order=list(rng.permutation(feats)), seed=seed)
             for _ in range(n_random)]
        rand_curves.append(np.mean(r, axis=0))
        rand_auc.append(np.mean([np.trapezoid(rc) / len(rc) for rc in r]))
    return {"shap_deletion_auc": float(np.mean(shap_auc)),
            "random_deletion_auc": float(np.mean(rand_auc)),
            "mean_curve_shap": np.mean(curves, axis=0),
            "mean_curve_random": np.mean(rand_curves, axis=0),
            "win_rate": float(np.mean(np.array(shap_auc) < np.array(rand_auc)))}


def agreement_report(per_customer: list[dict[str, LocalExplanation]], k: int = 5) -> pd.DataFrame:
    """Average top-k overlap between every pair of methods."""
    methods = list(per_customer[0].keys())
    table = pd.DataFrame(1.0, index=methods, columns=methods)
    for i, a in enumerate(methods):
        for b in methods[i + 1:]:
            v = np.mean([rank_agreement(c[a].contributions, c[b].contributions, k) for c in per_customer])
            table.loc[a, b] = table.loc[b, a] = v
    return table


def ground_truth_recovery(global_importance: pd.Series, truth=SIMULATED_TRUE_DRIVERS) -> dict:
    """Of the k most important features (k = #true drivers), how many were planted?"""
    k = len(truth)
    ranking = list(global_importance.sort_values(ascending=False).index)
    top = ranking[:k]
    demo_ranks = [ranking.index(f) + 1 for f in DEMOGRAPHICS if f in ranking]
    return {"k": k, "top_k": top, "hits": sorted(set(top) & set(truth)),
            "recall": len(set(top) & set(truth)) / k,
            "best_demographic_rank": min(demo_ranks), "n_features": len(ranking)}
