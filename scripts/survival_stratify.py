"""Compute survival stratification results for given models/cohorts/methods.

Outputs a CSV with per-model, per-cohort, per-risk-score, per-cutoff survival
statistics (n_high/low, median RFS, HR, log-rank p, C-index, AUROC).

Example:
    python scripts/survival_stratify.py \
        --models 9109a6c2 1361bef2 06c598c0 \
        --cohorts soramic lusanne \
        --risk-scores a b \
        --cutoffs median_frozen kmeans_within kmeans_log_within youden_frozen \
        --heads lr rf \
        --output results/eval/survival/stratify_results.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hcc_multimodal.survival import ALL_MODELS, COHORTS
from hcc_multimodal.survival.analysis import analyze_groups, concordance, is_balanced
from hcc_multimodal.survival.cutoffs import CUTOFF_METHODS
from hcc_multimodal.survival.data import load_source_aligned
from hcc_multimodal.survival.risk_scores import route_a_scores, route_b_scores


def _auroc(scores: pd.Series, rfs_2year: pd.Series) -> float | None:
    y = rfs_2year.dropna()
    common = scores.index.intersection(y.index)
    y = y.loc[common].astype(int)
    if y.nunique() < 2 or len(common) < 5:
        return None
    return float(roc_auc_score(y, scores.loc[common]))


def run(
    models: list[str],
    cohorts: list[str],
    risk_scores: list[str],
    cutoffs: list[str],
    heads: list[str],
    best_head_only: bool = False,
) -> pd.DataFrame:
    rows: list[dict] = []

    for model in models:
        train = load_source_aligned(model, "resection")
        for cohort in cohorts:
            test = load_source_aligned(model, cohort)
            for rs in risk_scores:
                if rs == "a":
                    for head in heads:
                        oof, test_scores = route_a_scores(head, train, test.X)
                        auroc = _auroc(test_scores, test.rfs_2year)
                        cidx = concordance(test_scores, test.time, test.event)
                        for cutoff_name in cutoffs:
                            fn = CUTOFF_METHODS[cutoff_name]
                            groups, _ = fn(oof, train.rfs_2year, test_scores)
                            _append_row(rows, model, head, cohort, "2yr_classifier",
                                        cutoff_name, groups, test, cidx, auroc)
                elif rs == "b":
                    oof, test_scores = route_b_scores(train, test.X)
                    cidx = concordance(test_scores, test.time, test.event)
                    auroc = _auroc(test_scores, test.rfs_2year)
                    for cutoff_name in cutoffs:
                        fn = CUTOFF_METHODS[cutoff_name]
                        groups, _ = fn(oof, train.rfs_2year, test_scores)
                        _append_row(rows, model, "cox", cohort, "cox_model",
                                    cutoff_name, groups, test, cidx, auroc)
        print(f"  done: {model}")

    df = pd.DataFrame(rows)
    if best_head_only and "head" in df.columns:
        idx = df.groupby(["model_id", "cohort", "risk_score", "cutoff"])["auroc"].idxmax()
        df = df.loc[idx.dropna()].reset_index(drop=True)
    return df


def _append_row(rows, model, head, cohort, risk_score, cutoff_name, groups, test, cidx, auroc):
    n_high = int((groups == "high").sum())
    n_low = int((groups == "low").sum())
    balanced = is_balanced(n_high, n_low)

    row = {
        "model_id": model,
        "head": head,
        "cohort": cohort,
        "risk_score": risk_score,
        "cutoff": cutoff_name,
        "n_high": n_high,
        "n_low": n_low,
        "balanced": balanced,
        "c_index": cidx,
        "auroc": auroc,
        "hr": None,
        "hr_lo": None,
        "hr_hi": None,
        "logrank_p": None,
        "median_rfs_high": None,
        "median_rfs_low": None,
    }

    if balanced:
        stats = analyze_groups(groups, test.time, test.event)
        row["hr"] = stats["hr_high_vs_low"]
        row["hr_lo"] = stats["hr_ci_low"]
        row["hr_hi"] = stats["hr_ci_high"]
        row["logrank_p"] = stats["logrank_p"]
        if stats.get("high"):
            row["median_rfs_high"] = stats["high"]["median_rfs"]
        if stats.get("low"):
            row["median_rfs_low"] = stats["low"]["median_rfs"]

    rows.append(row)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=list(ALL_MODELS),
                   help="Model IDs to evaluate (default: all 17)")
    p.add_argument("--cohorts", nargs="+", default=list(COHORTS),
                   help="Test cohorts (default: soramic lusanne)")
    p.add_argument("--risk-scores", nargs="+", default=["a", "b"], dest="risk_scores",
                   help="Risk score routes: a (2yr classifier), b (Cox)")
    p.add_argument("--cutoffs", nargs="+",
                   default=["median_frozen", "kmeans_within", "kmeans_log_within", "youden_frozen"],
                   help="Cutoff strategies to apply")
    p.add_argument("--heads", nargs="+", default=["lr", "rf"],
                   help="Classifier heads for Route A")
    p.add_argument("--best-head-only", action="store_true",
                   help="Keep only the best head (by AUROC) per model/cohort/route/cutoff")
    p.add_argument("--output", type=Path,
                   default=Path("results/eval/survival/stratify_results.csv"),
                   help="Output CSV path")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"Models: {args.models}")
    print(f"Cohorts: {args.cohorts}")
    print(f"Risk scores: {args.risk_scores}")
    print(f"Cutoffs: {args.cutoffs}")
    print(f"Heads: {args.heads}")

    df = run(args.models, args.cohorts, args.risk_scores, args.cutoffs, args.heads,
             best_head_only=args.best_head_only)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nWrote {len(df)} rows → {args.output}")
