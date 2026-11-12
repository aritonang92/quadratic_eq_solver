"""LIME - fit a tiny, honest straight line around one customer."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .base import FeynmanCard, LocalExplanation, mix_rows


class LimeExplainer:
    FEYNMAN = FeynmanCard(
        concept="LIME (Local Interpretable Model-agnostic Explanations)",
        simple=("Make thousands of slightly-different copies of this customer, where some "
                "facts are swapped for an ordinary customer's facts. Ask the ANN about each "
                "copy. Then fit a simple straight-line rule that mimics the ANN *only near "
                "this customer*. The line's slopes are the explanation."),
        analogy=("The Earth is round, but your street looks flat. LIME draws the flat street "
                 "map: wrong globally, good enough locally."),
        gaps=("Results depend on how 'near' is defined (kernel width) and on random "
              "sampling - run it twice and the numbers wiggle. If the local fit (R^2) is "
              "poor, the straight line is lying and the explanation should not be trusted."),
        refined=("Sample binary masks z in {0,1}^d; build x' = x where z=1, background where z=0; "
                 "weight each sample by pi(z) = exp(-(1 - mean z)^2 / w^2); solve weighted ridge "
                 "f(x') ~ b0 + sum_j beta_j z_j. beta_j = local effect of 'having this customer's "
                 "value of feature j'."),
    )

    def __init__(self, predict_fn, background: pd.DataFrame, n_samples: int = 3000,
                 kernel_width: float = 0.5, seed: int = 0):
        self.f = predict_fn
        self.background = background.reset_index(drop=True)
        self.n = n_samples
        self.width = kernel_width
        self.rng = np.random.default_rng(seed)

    def explain(self, x: pd.Series) -> LocalExplanation:
        d = len(self.background.columns)
        z = self.rng.random((self.n, d)) < self.rng.uniform(0.2, 1.0, (self.n, 1))
        z[0] = True                                            # include the customer itself
        bg = self.background.iloc[self.rng.integers(0, len(self.background), self.n)]
        y = self.f(mix_rows(x, bg.reset_index(drop=True), z))

        distance = 1.0 - z.mean(axis=1)                         # fraction of facts swapped
        w = np.exp(-(distance ** 2) / self.width ** 2)
        lin = Ridge(alpha=1e-2).fit(z.astype(float), y, sample_weight=w)
        r2 = lin.score(z.astype(float), y, sample_weight=w)
        return LocalExplanation(
            method="lime",
            customer=x,
            contributions=pd.Series(lin.coef_, index=self.background.columns),
            base_value=float(lin.intercept_),
            prediction=float(y[0]),
            diagnostics={"local_r2": float(r2)},
        )
