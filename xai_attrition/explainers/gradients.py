"""Integrated Gradients - looking *inside* the network rather than poking it."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import FEATURES, encoded_to_raw_map
from .base import FeynmanCard, LocalExplanation


class IntegratedGradientsExplainer:
    FEYNMAN = FeynmanCard(
        concept="Integrated Gradients",
        simple=("Slide a dial from 'typical customer' to 'this customer' in small steps. At "
                "each step ask the network: if I nudge each fact a tiny bit, how much does "
                "your score move? Add up those nudges along the whole slide. Each fact's "
                "total is its share of the difference."),
        analogy=("Hiking from a valley (typical customer) to a peak (this customer). Your "
                 "total climb splits into how much you climbed while walking north versus "
                 "east. Add up each direction's climb along the path."),
        gaps=("The answer depends on the starting point (baseline). 'Typical customer' is "
              "a choice, not a law. It works in the network's internal, scaled one-hot "
              "space, so we fold the one-hot pieces back into readable features."),
        refined=("IG_j(x) = (x_j - x'_j) * integral_0^1 dF(x' + a(x - x'))/dx_j da, computed on the "
                 "logit F with a midpoint Riemann sum; gradients come from hand-written back-"
                 "propagation. Completeness: sum_j IG_j = F(x) - F(x'). Units are log-odds."),
    )

    def __init__(self, model, background: pd.DataFrame, steps: int = 128):
        self.model = model
        self.pre = model.preprocessor
        self.baseline = self.pre.transform(background).mean(axis=0, keepdims=True)
        self.owners = np.array(encoded_to_raw_map(self.pre))
        self.steps = steps

    def explain(self, x: pd.Series) -> LocalExplanation:
        z = self.pre.transform(x.to_frame().T[FEATURES])
        alphas = (np.arange(self.steps) + 0.5) / self.steps
        path = self.baseline + alphas[:, None] * (z - self.baseline)
        _, grads = self.model.logit_and_grad(path)
        attr_encoded = (z - self.baseline).ravel() * grads.mean(axis=0)

        attr = pd.Series(attr_encoded).groupby(self.owners).sum().reindex(FEATURES)
        (fx,), _ = self.model.logit_and_grad(z)
        (f0,), _ = self.model.logit_and_grad(self.baseline)
        return LocalExplanation(
            method="integrated_gradients",
            customer=x,
            contributions=attr,
            base_value=float(f0),
            prediction=float(fx),
            unit="log-odds",
            diagnostics={"completeness_gap": float(f0 + attr.sum() - fx)},
        )
