"""Charts. Colour has one job each: red = pushes toward LEAVING, blue = toward STAYING."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from .data import CATEGORICAL_FEATURES, plain
from .explainers.base import LocalExplanation
from .explainers.global_methods import DependenceCurve

RISK_UP, RISK_DOWN = "#e34948", "#2a78d6"         # diverging poles
SERIES_2 = "#eb6834"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
SURFACE, GRID, AXIS = "#fcfcfb", "#e1e0d9", "#c3c2b7"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#6da7ec", "#256abf", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "font.size": 10, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
    "font.family": "sans-serif",
})


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def permutation_bar(imp: pd.DataFrame, path: Path) -> Path:
    imp = imp.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh([plain(f) for f in imp.feature], imp.importance, xerr=imp["std"], color=RISK_DOWN,
            height=0.6, error_kw={"ecolor": INK_2, "lw": 1})
    ax.set_xlabel("drop in ROC-AUC when the column is shuffled")
    ax.set_title("Which facts does the ANN rely on? (permutation importance)")
    ax.grid(axis="y", visible=False)
    return _save(fig, path)


def shap_summary(S: pd.DataFrame, X: pd.DataFrame, path: Path, top: int = 12) -> Path:
    order = S.abs().mean().sort_values(ascending=False).index[:top][::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    rng = np.random.default_rng(0)
    for i, f in enumerate(order):
        vals = S[f].to_numpy()
        jitter = rng.normal(0, 0.12, len(vals))
        if f in CATEGORICAL_FEATURES:
            ax.scatter(vals, i + jitter, s=10, color=MUTED, alpha=0.7, lw=0)
        else:
            raw = X[f].astype(float).to_numpy()
            norm = (raw - np.nanpercentile(raw, 5)) / (np.ptp(np.nanpercentile(raw, [5, 95])) + 1e-9)
            ax.scatter(vals, i + jitter, s=10, c=np.clip(norm, 0, 1), cmap=SEQ, lw=0)
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_yticks(range(len(order)), [plain(f) for f in order])
    ax.set_xlabel("Shapley value: change in P(leave)  (left = keeps customer, right = pushes out)")
    ax.set_title("How each fact moves risk, customer by customer")
    ax.grid(axis="y", visible=False)
    sm = plt.cm.ScalarMappable(cmap=SEQ)
    cb = fig.colorbar(sm, ax=ax, pad=0.01, aspect=40)
    cb.set_ticks([0, 1], labels=["low", "high"])
    cb.set_label("feature value (grey = category)", color=INK_2)
    cb.outline.set_visible(False)
    return _save(fig, path)


def waterfall(e: LocalExplanation, customer_id, path: Path, top: int = 8) -> Path:
    c = e.top(top)
    other = e.contributions.drop(c.index).sum()
    items = list(c.items()) + [("all other facts", other)]
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(items) + 1.6))
    level = e.base_value
    for i, (f, v) in enumerate(items):
        ax.barh(i, v, left=level, color=RISK_UP if v > 0 else RISK_DOWN, height=0.6)
        ax.text(max(level, level + v) + 0.005, i, f"{v * 100:+.1f} pts", va="center", color=INK_2, fontsize=9)
        level += v
    labels = [plain(f) if f in e.customer.index else f for f, _ in items]
    labels = [f"{lab} = {e.customer[f]:.4g}" if f in e.customer.index and f not in CATEGORICAL_FEATURES
              else f"{lab} = {e.customer[f]}" if f in e.customer.index else lab
              for lab, (f, _) in zip(labels, items)]
    ax.set_yticks(range(len(items)), labels)
    ax.invert_yaxis()
    ax.axvline(e.base_value, color=MUTED, lw=1, ls="--")
    ax.axvline(e.prediction, color=INK, lw=1.5)
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo, hi + 0.12 * (hi - lo))                   # room for the value labels
    ax.set_xlabel("P(leave)   (dashed: typical customer, solid: this customer)")
    ax.set_title(f"Customer {customer_id}: from typical {e.base_value:.0%} to {e.prediction:.0%}")
    ax.grid(axis="y", visible=False)
    return _save(fig, path)


def dependence_grid(curves: list[DependenceCurve], path: Path, n_ice: int = 60) -> Path:
    n = len(curves)
    cols = 2
    fig, axes = plt.subplots((n + 1) // cols, cols, figsize=(9, 3.2 * ((n + 1) // cols)), squeeze=False)
    rng = np.random.default_rng(0)
    for ax, c in zip(axes.ravel(), curves):
        x = np.arange(len(c.grid)) if c.categorical else c.grid
        idx = rng.choice(len(c.ice), min(n_ice, len(c.ice)), replace=False)
        for i in idx:
            ax.plot(x, c.ice[i], color=MUTED, alpha=0.25, lw=0.8)
        ax.plot(x, c.pdp, color=RISK_DOWN, lw=2.5, marker="o" if c.categorical else None)
        if c.categorical:
            ax.set_xticks(x, c.grid, rotation=30, ha="right", fontsize=8)
        ax.set_title(plain(c.feature), fontsize=10)
        ax.set_ylabel("P(leave)")
        ax.set_ylim(0, 1)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    fig.suptitle("Turning one dial at a time - grey: individual customers (ICE), blue: average (PDP)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    return _save(fig, path)


def deletion_plot(shap_curve: np.ndarray, random_curve: np.ndarray, path: Path) -> Path:
    k = np.arange(len(shap_curve))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(k, shap_curve, color=RISK_DOWN, lw=2, label="erase in SHAP order")
    ax.plot(k, random_curve, color=SERIES_2, lw=2, label="erase in random order")
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_xlabel("number of facts erased (replaced by an ordinary customer's)")
    ax.set_ylabel("mean P(leave) of high-risk customers")
    ax.set_title("Faithfulness: erasing what SHAP blames drops risk fastest")
    ax.legend(frameon=False)
    return _save(fig, path)
