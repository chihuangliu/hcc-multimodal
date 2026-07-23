"""Task 4 — screen all models (+ radiomic baseline) for AUC AND survival.

Goal: find a risk-score source that both classifies 2-year RFS well (AUROC) and
stratifies survival (C-index / HR), beating the radiomic baseline. Uses the
Route-A risk score (the classifier probability, which is what AUROC measures) for
every contrastive model and for the pre-trained radiomic pipeline, and tries the
three leakage-free within-cohort split variants.

Writes a long-format table: results/eval/survival/screen.csv (+ .json).

Run:  python -m hcc_multimodal.survival.screen
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime

import pandas as pd
from sklearn.metrics import roc_auc_score

from hcc_multimodal.eval.data import PROJECT_ROOT

from . import ALL_MODELS, COHORTS
from .analysis import analyze_groups, concordance, is_balanced
from .cutoffs import kmeans_log_within, kmeans_within, median_within
from .data import MODEL_INPUT, load_source_aligned
from .risk_scores import route_a_scores

SURVIVAL_DIR = PROJECT_ROOT / "results" / "eval" / "survival"
HEADS = ("lr", "rf")
SPLITS = {
    "median_within": median_within,
    "kmeans_within": kmeans_within,
    "kmeans_log_within": kmeans_log_within,
}


def _auroc(test_scores: pd.Series, rfs_2year: pd.Series) -> float | None:
    y = rfs_2year.dropna()
    common = test_scores.index.intersection(y.index)
    y = y.loc[common].astype(int)
    if y.nunique() < 2:
        return None
    return float(roc_auc_score(y, test_scores.loc[common]))


def _screen_one(source: str, head: str, train, test, rows: list) -> None:
    oof, test_scores = route_a_scores(head, train, test.X)
    auroc = _auroc(test_scores, test.rfs_2year)
    cidx = concordance(test_scores, test.time, test.event)
    base = {
        "source": source,
        "model_id": source if source != "radiomic" else "—",
        "input": MODEL_INPUT.get(source, "radiomic"),
        "head": head,
        "cohort": test.cohort,
        "n": int(len(test.X)),
        "auroc": auroc,
        "c_index": cidx,
    }
    for split_name, fn in SPLITS.items():
        groups, _ = fn(oof, train.rfs_2year, test_scores)
        n_high = int((groups == "high").sum())
        n_low = int((groups == "low").sum())
        row = {**base, "split": split_name, "n_high": n_high, "n_low": n_low,
               "hr": None, "hr_lo": None, "hr_hi": None, "logrank_p": None,
               "balanced": is_balanced(n_high, n_low)}
        if row["balanced"]:
            s = analyze_groups(groups, test.time, test.event)
            row.update(hr=s["hr_high_vs_low"], hr_lo=s["hr_ci_low"],
                       hr_hi=s["hr_ci_high"], logrank_p=s["logrank_p"])
        rows.append(row)


def main(args: argparse.Namespace) -> None:
    sources = ["radiomic", *ALL_MODELS]
    rows: list[dict] = []
    for source in sources:
        train = load_source_aligned(source, "resection")
        for cohort in COHORTS:
            test = load_source_aligned(source, cohort)
            for head in HEADS:
                _screen_one(source, head, train, test, rows)
        print(f"  done: {source}")

    df = pd.DataFrame(rows)
    SURVIVAL_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(SURVIVAL_DIR / "screen.csv", index=False)
    meta = {"timestamp": datetime.now().isoformat(), "n_rows": len(df)}
    (SURVIVAL_DIR / "screen.json").write_text(json.dumps(meta, indent=2))
    print(f"\nScreen table → {SURVIVAL_DIR / 'screen.csv'} ({len(df)} rows)")
    _print_top(df)


def _print_top(df: pd.DataFrame) -> None:
    """Per cohort, rank by AUROC and show C-index + best within-median HR/p."""
    for cohort in COHORTS:
        sub = df[(df.cohort == cohort) & (df.split == "median_within")].copy()
        sub = sub.sort_values("auroc", ascending=False)
        print(f"\n=== {cohort} (ranked by AUROC; median_within HR/p) ===")
        for _, r in sub.head(8).iterrows():
            a = "n/a" if pd.isna(r["auroc"]) else f"{r['auroc']:.3f}"
            c = "n/a" if pd.isna(r["c_index"]) else f"{r['c_index']:.3f}"
            hr = "n/a" if pd.isna(r["hr"]) else f"{r['hr']:.2f}"
            p = "n/a" if pd.isna(r["logrank_p"]) else f"{r['logrank_p']:.3f}"
            print(f"  {r['source']:10s} {r['head']} | AUROC={a} C={c} HR={hr:>5} p={p}")


def _parse() -> argparse.Namespace:
    return argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    ).parse_args()


if __name__ == "__main__":
    main(_parse())
