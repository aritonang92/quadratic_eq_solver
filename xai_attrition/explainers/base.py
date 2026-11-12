"""Shared vocabulary for every explainer.

Every explainer carries a ``FEYNMAN`` card - the four steps of the Feynman
technique applied to itself:

1. concept   - name the thing.
2. simple    - explain it so a 12-year-old gets it.
3. gaps      - where the simple story breaks (go back and study those).
4. refined   - the precise version, now that the intuition is in place.

The report prints these cards, so a reader learns *what* each chart means before
reading it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeynmanCard:
    concept: str
    simple: str
    analogy: str
    gaps: str
    refined: str

    def to_markdown(self) -> str:
        return (
            f"**{self.concept}**\n\n"
            f"- *In plain words:* {self.simple}\n"
            f"- *Analogy:* {self.analogy}\n"
            f"- *Where the simple story breaks:* {self.gaps}\n"
            f"- *Precisely:* {self.refined}\n"
        )


@dataclass
class LocalExplanation:
    """How much each raw feature pushed ONE customer's prediction, and from where."""

    method: str
    customer: pd.Series             # the raw feature values
    contributions: pd.Series        # feature -> push (units given by ``unit``)
    base_value: float               # where the story starts (typical customer)
    prediction: float               # where it ends (this customer)
    unit: str = "probability"
    diagnostics: dict = field(default_factory=dict)

    def top(self, k: int = 5) -> pd.Series:
        return self.contributions.reindex(
            self.contributions.abs().sort_values(ascending=False).index).head(k)


def rank_agreement(a: pd.Series, b: pd.Series, k: int = 5) -> float:
    """Share of the top-k features (by |value|) that two explanations have in common."""
    ta = set(a.abs().sort_values(ascending=False).index[:k])
    tb = set(b.abs().sort_values(ascending=False).index[:k])
    return len(ta & tb) / k


def mix_rows(x: pd.Series, background: pd.DataFrame, keep_mask: np.ndarray) -> pd.DataFrame:
    """Build hybrid customers: where ``keep_mask`` is True use x's value, else background's.

    keep_mask has shape (n_rows, n_features); background must have n_rows rows.
    This one helper is the engine behind Shapley and LIME: "what if this customer
    had *their own* value for these features, and an ordinary customer's value
    for the rest?"
    """
    cols = {}
    for j, col in enumerate(background.columns):
        bg = background[col].to_numpy()
        cols[col] = np.where(keep_mask[:, j], x[col], bg)
        if pd.api.types.is_numeric_dtype(background[col]):
            cols[col] = cols[col].astype(float)
    return pd.DataFrame(cols, columns=background.columns)
