"""Shapley values - fair credit-sharing for a single prediction."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import FeynmanCard, LocalExplanation, mix_rows


class ShapleyExplainer:
    FEYNMAN = FeynmanCard(
        concept="Shapley values (SHAP)",
        simple=("Start from the risk of a typical customer. Add this customer's facts one "
                "at a time and watch the risk move. A fact's credit is how much it moved "
                "the risk - averaged over every possible order of adding the facts, so "
                "nobody gets extra credit just for arriving first."),
        analogy=("A team wins a prize. To split it fairly you ask, for every possible "
                 "line-up order: how much better did the team get when this player joined? "
                 "Their fair share is the average of those jumps."),
        gaps=("'Remove a fact' really means 'replace it with a value from a random other "
              "customer', which can create odd combinations (e.g. huge spend but 10 "
              "transactions). Correlated features share credit. And the result says what "
              "the MODEL uses, not what CAUSES customers to leave."),
        refined=("phi_j = E over random orderings pi and background rows z of "
                 "[f(x_{S u j}, z_rest) - f(x_S, z_rest)], with S the features before j in pi. "
                 "Estimated by Monte-Carlo with antithetic (reversed) orderings. By "
                 "construction sum_j phi_j = f(x) - E[f(z)] (efficiency), so the pushes "
                 "add up exactly to the prediction; the small Monte-Carlo gap to the exact "
                 "background mean is spread over features by their sampling variance."),
    )

    def __init__(self, predict_fn, background: pd.DataFrame, n_permutations: int = 128, seed: int = 0):
        self.f = predict_fn
        self.background = background.reset_index(drop=True)
        self.n_perm = n_permutations + (n_permutations % 2)  # even, for antithetic pairs
        self.rng = np.random.default_rng(seed)
        self.base_value = float(np.mean(self.f(self.background)))   # the "typical customer"

    def explain(self, x: pd.Series) -> LocalExplanation:
        d = len(self.background.columns)
        half = self.n_perm // 2
        perms = np.array([self.rng.permutation(d) for _ in range(half)])
        perms = np.vstack([perms, perms[:, ::-1]])           # antithetic pairs reduce variance
        M = len(perms)
        rank = np.argsort(perms, axis=1)                     # rank[m, f] = position of f in perm m

        # Step k of chain m keeps the first k features of perm m from x.
        steps = np.arange(d + 1)
        keep = rank[:, None, :] < steps[None, :, None]       # (M, d+1, d)
        bg_idx = np.resize(self.rng.permutation(len(self.background)), M)  # spread over background
        bg = self.background.iloc[np.repeat(bg_idx, d + 1)].reset_index(drop=True)

        preds = self.f(mix_rows(x, bg, keep.reshape(-1, d))).reshape(M, d + 1)
        jumps = np.take_along_axis(preds, rank + 1, axis=1) - np.take_along_axis(preds, rank, axis=1)

        phi = jumps.mean(axis=0)
        stderr = jumps.std(axis=0, ddof=1) / np.sqrt(M)
        fx = preds[:, -1].mean()
        # Monte-Carlo leaves a tiny gap between the sampled start and the exact average
        # customer. Spread it over features in proportion to their sampling noise, so the
        # pushes add up exactly from the SAME starting point for every customer.
        gap = (fx - self.base_value) - phi.sum()
        noise = stderr ** 2
        phi = phi + gap * (noise / noise.sum() if noise.sum() > 0 else 1 / d)
        base = self.base_value
        return LocalExplanation(
            method="shapley",
            customer=x,
            contributions=pd.Series(phi, index=self.background.columns),
            base_value=float(base),
            prediction=float(fx),
            diagnostics={"stderr": pd.Series(stderr, index=self.background.columns),
                         "additivity_gap": float(base + phi.sum() - fx)},
        )

    def explain_many(self, X: pd.DataFrame) -> pd.DataFrame:
        """Shapley matrix (rows = customers, cols = features) for global summaries."""
        return pd.DataFrame([self.explain(X.iloc[i]).contributions for i in range(len(X))],
                            index=X.index)
