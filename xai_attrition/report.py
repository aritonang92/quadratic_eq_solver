"""Write everything to ``outputs/``: charts, a Markdown report, and a JSON summary."""
from __future__ import annotations

import json
from pathlib import Path

from . import plots
from .data import plain
from .explainers.counterfactual import CounterfactualExplainer
from .explainers.global_methods import GlobalSurrogate, PartialDependence, PermutationImportance
from .explainers.gradients import IntegratedGradientsExplainer
from .explainers.lime import LimeExplainer
from .explainers.shapley import ShapleyExplainer
from .framework import XAIResults

CARDS = [PermutationImportance, ShapleyExplainer, PartialDependence, GlobalSurrogate,
         LimeExplainer, IntegratedGradientsExplainer, CounterfactualExplainer]


def build_report(r: XAIResults, out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    img = out / "figures"
    img.mkdir(exist_ok=True)

    plots.permutation_bar(r.permutation, img / "permutation_importance.png")
    plots.shap_summary(r.shap_matrix, r.shap_sample, img / "shap_summary.png")
    plots.dependence_grid(r.dependence, img / "pdp_ice.png")
    plots.deletion_plot(r.faithfulness["mean_curve_shap"], r.faithfulness["mean_curve_random"],
                        img / "faithfulness_deletion.png")
    for i, c in enumerate(r.cases):
        plots.waterfall(c.explanations["shapley"], c.customer_id, img / f"waterfall_case{i + 1}.png")

    m = r.metrics
    md = [
        "# Explainable AI report - credit-card attrition ANN",
        "",
        f"*Data:* {r.source}. *Base attrition rate:* {r.base_rate:.1%}.",
        "",
        "## 0. The black box, in one paragraph",
        "",
        "A neural network (2 hidden layers, 64 and 32 ReLU units) reads 18 facts about a "
        "cardholder and outputs the probability they will close their account. It is accurate "
        "but opaque: its knowledge is spread over thousands of weights. Below we open it in "
        "three passes - **global** (how it thinks overall), **local** (why this customer), and "
        "**trust** (whether the explanations themselves hold up).",
        "",
        "| metric (held-out) | value |", "|---|---|",
        f"| ROC-AUC | {m['roc_auc']:.3f} |",
        f"| PR-AUC | {m['pr_auc']:.3f} |",
        f"| F1 (attrition) | {m['f1']:.3f} |",
        f"| Recall of leavers | {m['recall_attrition']:.1%} |",
        f"| Precision on leavers | {m['precision_attrition']:.1%} |",
        f"| Confusion (tn/fp/fn/tp) | {m['confusion']['tn']}/{m['confusion']['fp']}/"
        f"{m['confusion']['fn']}/{m['confusion']['tp']} |",
        "",
        "## 1. Global - how does the ANN think?",
        "",
        PermutationImportance.FEYNMAN.to_markdown(),
        "![permutation](figures/permutation_importance.png)",
        "",
        ShapleyExplainer.FEYNMAN.to_markdown(),
        "![shap](figures/shap_summary.png)",
        "",
        PartialDependence.FEYNMAN.to_markdown(),
        "![pdp](figures/pdp_ice.png)",
        "",
        GlobalSurrogate.FEYNMAN.to_markdown(),
        f"The depth-{r.surrogate.tree.get_depth()} tree agrees with the ANN on "
        f"**{r.surrogate.fidelity:.1%}** of held-out customers (fidelity AUC "
        f"{r.surrogate.fidelity_auc:.3f}). The riskiest rules it found:",
        "",
        "| IF ... | customers | ANN says leave |", "|---|---|---|",
    ]
    for rule in r.surrogate.rules[:5]:
        md.append(f"| {' AND '.join(rule['conditions'])} | {rule['support']} | {rule['ann_attrition_rate']:.0%} |")

    md += ["", "## 2. Local - why THIS customer?", "",
           LimeExplainer.FEYNMAN.to_markdown(), IntegratedGradientsExplainer.FEYNMAN.to_markdown(),
           CounterfactualExplainer.FEYNMAN.to_markdown()]
    for i, c in enumerate(r.cases):
        md += ["", f"#### Case {i + 1}: {c.label} (actually {'left' if c.actual_left else 'stayed'})", "",
               f"![case{i + 1}](figures/waterfall_case{i + 1}.png)", "", c.story]

    f, g = r.faithfulness, r.ground_truth
    md += [
        "", "## 3. Trust - should we believe the explanations?", "",
        "![deletion](figures/faithfulness_deletion.png)", "",
        f"- **Faithfulness.** Erasing facts in SHAP's order lowers risk faster than random "
        f"order for **{f['win_rate']:.0%}** of the highest-risk customers (area under deletion curve "
        f"{f['shap_deletion_auc']:.3f} vs {f['random_deletion_auc']:.3f}; lower is better).",
        f"- **Internal consistency.** SHAP pushes add up to the prediction (max gap "
        f"{r.extras['shap_additivity_gap']:.1e}); Integrated Gradients satisfy completeness (max gap "
        f"{r.extras['ig_completeness_gap']:.1e} log-odds); LIME's mean local fit R^2 = {r.extras['lime_r2']:.2f}.",
        "- **Agreement between methods** (average share of top-5 drivers in common):", "",
        r.agreement.round(2).to_markdown(), "",
    ]
    if g:
        md += [f"- **Ground truth recovery.** We planted {g['k']} real drivers in the simulator. "
               f"The {g['k']} features SHAP ranks highest contain **{len(g['hits'])}/{g['k']}** of them "
               f"(recall {g['recall']:.0%}): {', '.join(plain(h) for h in g['hits'])}. "
               f"Demographics were planted as pure noise; the highest-ranked one sits at "
               f"#{g['best_demographic_rank']} of {g['n_features']}.", ""]
    md += ["## 4. Limits - what these explanations do NOT tell you", "",
           "- They describe the **model**, not the **world**. 'Few transactions raises risk' means "
           "the ANN uses it, not that forcing transactions will stop customers leaving.",
           "- Correlated facts (spend vs. transaction count) share credit; read them as a group.",
           "- Perturbation methods invent hybrid customers that may be unrealistic.",
           "- Every method has knobs (baseline, kernel width, sample size). The trust section "
           "exists so you can see when those knobs matter.", ""]

    path = out / "report.md"
    path.write_text("\n".join(md))
    summary = {
        "source": r.source, "metrics": r.metrics,
        "permutation_importance": r.permutation.to_dict(orient="records"),
        "mean_abs_shap": r.shap_matrix.abs().mean().sort_values(ascending=False).to_dict(),
        "surrogate_fidelity": r.surrogate.fidelity,
        "faithfulness": {k: v for k, v in f.items() if not k.startswith("mean_curve")},
        "agreement": r.agreement.to_dict(), "ground_truth": g, "checks": r.extras,
        "cases": [{"id": int(c.customer_id), "label": c.label, "actually_left": c.actual_left,
                   "p_leave": c.explanations["shapley"].prediction} for c in r.cases],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    return path
