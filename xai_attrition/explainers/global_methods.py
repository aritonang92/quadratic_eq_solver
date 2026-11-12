"""Global explanations - how the ANN behaves across ALL customers."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier, _tree

from ..data import CATEGORICAL_FEATURES, NUMERIC_FEATURES, plain
from .base import FeynmanCard


# --------------------------------------------------------------------------- #
# Permutation importance
# --------------------------------------------------------------------------- #
class PermutationImportance:
    FEYNMAN = FeynmanCard(
        concept="Permutation feature importance",
        simple=("Shuffle one column - say 'number of transactions' - so it becomes random "
                "noise, and see how much worse the ANN gets at spotting leavers. A big drop "
                "means the ANN leaned on that column heavily."),
        analogy=("To learn which ingredient matters in a recipe, swap the sugar for a random "
                 "pantry item and taste. If the cake is ruined, sugar mattered."),
        gaps=("If two columns carry the same information, shuffling one barely hurts (the "
              "other covers for it), so both look unimportant. It tells you HOW MUCH a "
              "feature matters, not in WHICH DIRECTION."),
        refined=("I_j = AUC(X, y) - E_shuffle[AUC(X with column j permuted, y)], repeated "
                 "n times on held-out data; error bars are the std over repeats."),
    )

    def __init__(self, predict_fn, n_repeats: int = 5, seed: int = 0):
        self.f, self.n_repeats = predict_fn, n_repeats
        self.rng = np.random.default_rng(seed)

    def compute(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        base = roc_auc_score(y, self.f(X))
        rows = []
        for col in X.columns:
            drops = []
            for _ in range(self.n_repeats):
                Xp = X.copy()
                Xp[col] = self.rng.permutation(Xp[col].to_numpy())
                drops.append(base - roc_auc_score(y, self.f(Xp)))
            rows.append({"feature": col, "importance": np.mean(drops), "std": np.std(drops)})
        return pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Partial dependence + ICE
# --------------------------------------------------------------------------- #
@dataclass
class DependenceCurve:
    feature: str
    grid: np.ndarray
    ice: np.ndarray          # (n_customers, n_grid): one line per customer
    pdp: np.ndarray          # average line

    @property
    def categorical(self) -> bool:
        return self.feature in CATEGORICAL_FEATURES


class PartialDependence:
    FEYNMAN = FeynmanCard(
        concept="Partial dependence (PDP) and Individual Conditional Expectation (ICE)",
        simple=("Take one customer and turn one dial - e.g. number of transactions - from "
                "low to high while freezing everything else. Plot the risk at each setting. "
                "That one line is ICE. Do it for many customers and average: that's the PDP, "
                "the 'shape' of the rule the ANN learned."),
        analogy=("Adjusting the thermostat in many different houses and recording how warm "
                 "each one gets. The average curve is the PDP; the spread of individual "
                 "houses (ICE) shows whether every house reacts the same way."),
        gaps=("Freezing other features can create impossible customers (high spend with "
              "very few transactions). If ICE lines cross or fan out, the feature interacts "
              "with others and the average curve hides that."),
        refined=("ICE_i(v) = f(x_i with x_ij := v); PD(v) = mean_i ICE_i(v) over a quantile grid "
                 "of feature j (or its categories)."),
    )

    def __init__(self, predict_fn, n_grid: int = 25):
        self.f, self.n_grid = predict_fn, n_grid

    def compute(self, X: pd.DataFrame, feature: str) -> DependenceCurve:
        if feature in CATEGORICAL_FEATURES:
            grid = np.array(sorted(X[feature].unique()))
        else:
            grid = np.unique(np.quantile(X[feature], np.linspace(0.02, 0.98, self.n_grid)))
        stacked = pd.concat([X] * len(grid), ignore_index=True)
        stacked[feature] = np.repeat(grid, len(X))
        ice = self.f(stacked).reshape(len(grid), len(X)).T
        return DependenceCurve(feature, grid, ice, ice.mean(axis=0))


# --------------------------------------------------------------------------- #
# Global surrogate tree
# --------------------------------------------------------------------------- #
@dataclass
class SurrogateResult:
    tree: DecisionTreeClassifier
    feature_names: list[str]
    fidelity: float            # how often the tree agrees with the ANN (held-out)
    fidelity_auc: float        # tree's AUC at predicting the ANN's labels
    rules: list[dict]          # leaf rules, riskiest first


class GlobalSurrogate:
    FEYNMAN = FeynmanCard(
        concept="Global surrogate decision tree",
        simple=("Train a small flow-chart (a decision tree with a few yes/no questions) to "
                "copy the ANN's answers. If the copy agrees with the ANN most of the time, "
                "the flow-chart is a readable summary of what the ANN does."),
        analogy=("A student writes a one-page cheat sheet that reproduces a professor's "
                 "grading. If the cheat sheet gives the same grade 95% of the time, it is a "
                 "good summary of the professor's mind."),
        gaps=("The tree explains the ANN only as well as its fidelity score. The remaining "
              "disagreement is exactly the part the cheat sheet cannot capture. A deeper "
              "tree is more faithful but less readable - a trade-off, not a free lunch."),
        refined=("Fit T = argmin_tree L(tree(X), 1[f(X) >= t]) with depth <= k; report "
                 "fidelity = mean(T(X_test) == ANN(X_test)) and extract each leaf's path as "
                 "an IF-THEN rule with its ANN-attrition rate and support."),
    )

    def __init__(self, model, max_depth: int = 4, min_samples_leaf: int = 40, seed: int = 0):
        self.model, self.max_depth, self.min_leaf, self.seed = model, max_depth, min_samples_leaf, seed

    @staticmethod
    def _encode(X: pd.DataFrame) -> pd.DataFrame:
        dummies = pd.get_dummies(X[CATEGORICAL_FEATURES], prefix_sep="=").astype(int)
        return pd.concat([X[NUMERIC_FEATURES], dummies], axis=1)

    def fit(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> SurrogateResult:
        Etr, Ete = self._encode(X_train), self._encode(X_test)
        Ete = Ete.reindex(columns=Etr.columns, fill_value=0)
        ytr, yte = self.model.predict(X_train), self.model.predict(X_test)
        tree = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_leaf=self.min_leaf,
                                      class_weight="balanced", random_state=self.seed).fit(Etr, ytr)
        fidelity = accuracy_score(yte, tree.predict(Ete))
        fid_auc = roc_auc_score(yte, tree.predict_proba(Ete)[:, 1]) if yte.min() != yte.max() else float("nan")
        rules = self._rules(tree, list(Etr.columns), Etr, ytr)
        return SurrogateResult(tree, list(Etr.columns), fidelity, fid_auc, rules)

    @staticmethod
    def _rules(tree, names, E, y_ann) -> list[dict]:
        t = tree.tree_
        leaf_of = tree.apply(E)
        out = []

        def walk(node, conds):
            if t.feature[node] == _tree.TREE_UNDEFINED:
                in_leaf = leaf_of == node
                out.append({"conditions": conds, "support": int(in_leaf.sum()),
                            "ann_attrition_rate": float(y_ann[in_leaf].mean()) if in_leaf.any() else 0.0})
                return
            name, thr = names[t.feature[node]], t.threshold[node]
            walk(t.children_left[node], conds + [(name, "<=", thr)])
            walk(t.children_right[node], conds + [(name, ">", thr)])

        walk(0, [])
        for r in out:
            r["conditions"] = _simplify(r["conditions"])
        return sorted(out, key=lambda r: -r["ann_attrition_rate"])


def _simplify(conds: list[tuple[str, str, float]]) -> list[str]:
    """Merge repeated tests on one feature ("x <= 57 AND x <= 46" -> "x <= 46")."""
    bounds: dict[str, list] = {}
    for name, op, thr in conds:
        lo, hi = bounds.setdefault(name, [-np.inf, np.inf])
        bounds[name] = [lo, min(hi, thr)] if op == "<=" else [max(lo, thr), hi]
    text = []
    for name, (lo, hi) in bounds.items():
        if "=" in name:                   # one-hot column: threshold 0.5 means is / is not
            feat, cat = name.split("=", 1)
            text.append(f"{plain(feat)} {'is' if lo >= 0.5 else 'is not'} {cat}")
        elif np.isfinite(lo) and np.isfinite(hi):
            text.append(f"{lo:,.2f} < {plain(name)} <= {hi:,.2f}")
        else:
            text.append(f"{plain(name)} > {lo:,.2f}" if np.isfinite(lo) else f"{plain(name)} <= {hi:,.2f}")
    return text
