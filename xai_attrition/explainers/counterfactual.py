"""Counterfactuals - 'what is the smallest change that flips the decision?'"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data import ACTIONABLE_FEATURES, FEATURES, recompute_derived
from .base import FeynmanCard


@dataclass
class Counterfactual:
    original: pd.Series
    counterfactual: pd.Series
    p_before: float
    p_after: float
    changes: list[tuple[str, float, float]]   # (feature, old, new)
    success: bool


class CounterfactualExplainer:
    FEYNMAN = FeynmanCard(
        concept="Counterfactual explanations",
        simple=("Instead of 'why', answer 'what would need to be different?'. Search for the "
                "smallest realistic change to things the bank can influence (usage, balance, "
                "products, contacts) that makes the ANN say 'this customer stays'."),
        analogy=("A loan officer saying 'you'd have been approved with $5k more income' - "
                 "the most useful explanation is often a recipe for a different outcome."),
        gaps=("It shows what changes the MODEL's mind, not what will change the CUSTOMER's. "
              "There are many possible counterfactuals; we return one greedy path. Only "
              "actionable features are touched - age or gender are never 'advice'."),
        refined=("Greedy search: at each step move one actionable feature one step along its "
                 "training-quantile grid, choosing the move with the largest drop in P(leave) "
                 "per unit of cost (|change| / IQR); stop once P(leave) < threshold."),
    )

    def __init__(self, model, X_reference: pd.DataFrame, actionable=ACTIONABLE_FEATURES,
                 n_grid: int = 41, max_steps: int = 40):
        self.model = model
        self.actionable = list(actionable)
        self.max_steps = max_steps
        qs = np.linspace(0, 1, n_grid)
        self.grids = {f: np.unique(np.quantile(X_reference[f], qs)) for f in self.actionable}
        self.scale = {f: max(np.subtract(*np.quantile(X_reference[f], [0.75, 0.25])), 1e-9)
                      for f in self.actionable}

    def explain(self, x: pd.Series) -> Counterfactual:
        target = self.model.threshold
        current = x[FEATURES].copy()
        p0 = p = float(self.model.predict_proba(current.to_frame().T)[0])
        for _ in range(self.max_steps):
            if p < target:
                break
            candidates = []
            for f in self.actionable:
                grid, cur = self.grids[f], float(current[f])
                below, above = grid[grid < cur], grid[grid > cur]
                neighbours = ([below.max()] if below.size else []) + ([above.min()] if above.size else [])
                for new in neighbours:
                    cand = current.copy()
                    cand[f] = new
                    candidates.append((f, cand))
            if not candidates:
                break
            frame = recompute_derived(pd.DataFrame([c for _, c in candidates]))
            probs = self.model.predict_proba(frame)
            gains = [(p - pn) / (abs(c[f] - x[f]) / self.scale[f] + 1e-9)
                     for (f, c), pn in zip(candidates, probs)]
            best = int(np.argmax(gains))
            if probs[best] >= p:
                break                                           # no move helps; give up
            current, p = frame.iloc[best], float(probs[best])

        changes = [(f, float(x[f]), float(current[f])) for f in self.actionable
                   if not np.isclose(float(x[f]), float(current[f]))]
        return Counterfactual(x, current, p0, p, changes, success=p < target)
