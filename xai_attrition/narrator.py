"""
Turn numbers into a story a relationship manager can read in 20 seconds.

FEYNMAN NOTE
------------
The final step of the Feynman technique is: if you can't say it simply, you don't
understand it yet. So every customer gets the same five-part story:

  1. The verdict          - risk %, in words (low / elevated / high).
  2. The starting point   - "a typical customer has X% risk".
  3. The pushes           - each fact that moved the risk, with its size and
                            compared to what is typical ("38 transactions vs 67").
  4. The what-if          - the smallest realistic change that flips the verdict.
  5. The confidence note  - do the methods agree, and does the local fit hold?
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import CATEGORICAL_FEATURES, plain
from .explainers.base import LocalExplanation, rank_agreement
from .explainers.counterfactual import Counterfactual


def risk_band(p: float) -> str:
    return "HIGH" if p >= 0.5 else "ELEVATED" if p >= 0.2 else "LOW"


def _fmt(feature: str, value) -> str:
    if feature in CATEGORICAL_FEATURES:
        return str(value)
    v = float(value)
    if feature in ("Credit_Limit", "Total_Revolving_Bal", "Total_Trans_Amt"):
        return f"${v:,.0f}"
    if feature == "Avg_Utilization_Ratio":
        return f"{v:.0%}"
    return f"{v:,.2f}".rstrip("0").rstrip(".")


def _fact(feature: str, value, typical: pd.Series) -> str:
    text = f"{plain(feature)} = {_fmt(feature, value)}"
    if feature not in CATEGORICAL_FEATURES:
        text += f" (typical: {_fmt(feature, typical[feature])})"
    return text


def narrate(customer_id, shap: LocalExplanation, lime: LocalExplanation,
            ig: LocalExplanation, cf: Counterfactual | None, typical: pd.Series, k: int = 4) -> str:
    p = shap.prediction
    lines = [f"### Customer {customer_id} - risk of leaving: {p:.0%} ({risk_band(p)})", ""]

    ups = shap.contributions[shap.contributions > 0].sort_values(ascending=False).head(k)
    downs = shap.contributions[shap.contributions < 0].sort_values().head(k)
    main = ups.index[0] if p >= shap.base_value and len(ups) else (downs.index[0] if len(downs) else None)
    if main is not None:
        verb = "mainly because of" if p >= shap.base_value else "mainly protected by"
        lines.append(f"**In one sentence:** the model rates this customer {risk_band(p).lower()} "
                     f"risk, {verb} their {_fact(main, shap.customer[main], typical)}.")
        lines.append("")

    lines.append(f"**The story:** a typical customer starts at **{shap.base_value:.0%}** risk. "
                 f"This customer's facts move it to **{p:.0%}**:")
    lines.append("")
    for f, v in ups.items():
        lines.append(f"- raises risk by {v * 100:+.1f} pts - {_fact(f, shap.customer[f], typical)}")
    for f, v in downs.items():
        lines.append(f"- lowers risk by {v * 100:+.1f} pts - {_fact(f, shap.customer[f], typical)}")
    rest = shap.contributions.drop(list(ups.index) + list(downs.index)).sum()
    lines.append(f"- everything else together: {rest * 100:+.1f} pts")
    lines.append("")

    if cf is not None and cf.p_before >= 0.5:
        if cf.success:
            moves = "; ".join(f"{plain(f)} {_fmt(f, a)} -> {_fmt(f, b)}" for f, a, b in cf.changes)
            lines.append(f"**What would change the outcome:** {moves} "
                         f"would bring the predicted risk to {cf.p_after:.0%} (retain).")
        else:
            lines.append("**What would change the outcome:** no small change to actionable "
                         f"levers flips the verdict (best found: {cf.p_after:.0%}). Escalate to a human.")
        lines.append("")

    agree_lime = rank_agreement(shap.contributions, lime.contributions, 3)
    agree_ig = rank_agreement(shap.contributions, ig.contributions, 3)
    r2 = lime.diagnostics.get("local_r2", np.nan)
    confidence = "high" if min(agree_lime, agree_ig) >= 2 / 3 and r2 >= 0.5 else \
                 "moderate" if max(agree_lime, agree_ig) >= 2 / 3 else "low"
    lines.append(f"**How much to trust this explanation:** {confidence}. Top-3 drivers shared "
                 f"with LIME: {agree_lime * 3:.0f}/3, with Integrated Gradients: {agree_ig * 3:.0f}/3; "
                 f"LIME local fit R^2 = {r2:.2f}.")
    return "\n".join(lines)
