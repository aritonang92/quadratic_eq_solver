"""Run the whole framework:  python -m xai_attrition [--csv BankChurners.csv] [--out outputs] [--fast]"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from .framework import AttritionXAI
from .report import build_report


def main() -> None:
    ap = argparse.ArgumentParser(description="Explainable-AI framework for cardholder attrition ANNs")
    ap.add_argument("--csv", help="Path to Kaggle BankChurners.csv (optional; simulated data otherwise)")
    ap.add_argument("--out", default="outputs", help="Output folder for report and figures")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--fast", action="store_true", help="Smaller samples - for a quick smoke run")
    args = ap.parse_args()

    t0 = time.time()
    print("1/3  Loading data and training the ANN ...")
    xai = AttritionXAI(csv_path=args.csv, seed=args.seed, fast=args.fast).fit()
    print(f"     source: {xai.data.source}")
    print("2/3  Explaining (global -> local -> trust) ...")
    results = xai.run()
    m = results.metrics
    print(f"     ROC-AUC {m['roc_auc']:.3f} | PR-AUC {m['pr_auc']:.3f} | F1 {m['f1']:.3f}")
    print("3/3  Writing report ...")
    path = build_report(results, Path(args.out))
    print(f"\nDone in {time.time() - t0:.0f}s -> {path}\n")
    print(results.cases[0].story)


if __name__ == "__main__":
    main()
