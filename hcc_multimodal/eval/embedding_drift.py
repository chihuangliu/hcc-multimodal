"""KS-test distribution drift analysis for contrastive embeddings.

Compares patient-level embedding distributions across three cohorts:
  resection (training) · soramic (ablation test) · lausanne (external test)

For each model and each cohort pair, runs a per-dimension KS test and reports:
  median_d  – median KS D-statistic across embedding dimensions
  mean_d    – mean   KS D-statistic across embedding dimensions
  frac_sig  – fraction of dimensions with p < 0.05

Usage:
  python -m hcc_multimodal.eval.embedding_drift [--out PATH]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from hcc_multimodal.eval.data import TRAINING_ROOT
from hcc_multimodal.eval.eval_utils import PROJECT_ROOT

# (model_id, input_suffix)
MODEL_CONFIGS: list[tuple[str, str]] = [
    # Group 1 — raw, 40 genes, n=10
    ("6a1a1bdf", "raw"),
    ("1361bef2", "raw"),
    ("982a6fa2", "raw"),
    ("a6f970d6", "raw"),
    # Group 2 — raw, gene-set ablation
    ("12e4ba6a", "raw"),
    ("34e6806f", "raw"),
    ("5d04e6ba", "raw"),
    ("9109a6c2", "raw"),
    # Group 3 — raw, frozen, n=all
    ("dc7e1d10", "raw"),
    ("5e3f71a0", "raw"),
    ("a64b245f", "raw"),
    ("06c598c0", "raw"),
    # Group 4 — bbox
    ("050d401d", "bbox"),
    ("f8aabb75", "bbox"),
    ("e12b0592", "bbox"),
    ("8715461c", "bbox"),
    ("92b9afed", "bbox"),
]

COMPARISONS = [
    ("resection", "soramic"),
    ("resection", "lausanne"),
    ("soramic",   "lausanne"),
]


_COHORT_FILENAME = {"soramic": "soramic", "lausanne": "lusanne"}


def _emb_path(model_id: str, suffix: str, cohort: str) -> Path:
    base = TRAINING_ROOT / model_id / "cached_embeddings"
    if cohort == "resection":
        return base / "resection_img_emb.parquet"
    return base / f"ablation_{_COHORT_FILENAME[cohort]}_img_emb_{suffix}.parquet"


def ks_drift(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    n_dims = a.shape[1]
    d_stats = np.empty(n_dims)
    p_vals  = np.empty(n_dims)
    for i in range(n_dims):
        res = ks_2samp(a[:, i], b[:, i])
        d_stats[i] = res.statistic
        p_vals[i]  = res.pvalue
    return {
        "median_d": float(np.median(d_stats)),
        "mean_d":   float(np.mean(d_stats)),
        "frac_sig": float(np.mean(p_vals < 0.05)),
    }


def run(out_path: Path) -> pd.DataFrame:
    records = []
    for model_id, suffix in MODEL_CONFIGS:
        embs: dict[str, np.ndarray] = {}
        for cohort in ("resection", "soramic", "lausanne"):
            embs[cohort] = pd.read_parquet(_emb_path(model_id, suffix, cohort)).values

        for cohort_a, cohort_b in COMPARISONS:
            stats = ks_drift(embs[cohort_a], embs[cohort_b])
            records.append(
                {
                    "model_id":   model_id,
                    "input":      suffix,
                    "comparison": f"{cohort_a}_vs_{cohort_b}",
                    **stats,
                }
            )

    df = pd.DataFrame(records)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved → {out_path}")
    return df


def _print_summary(df: pd.DataFrame) -> None:
    for comparison, grp in df.groupby("comparison"):
        print(f"\n{'='*60}")
        print(f"  {comparison}")
        print(f"{'='*60}")
        show = grp[["model_id", "input", "median_d", "mean_d", "frac_sig"]].copy()
        print(show.sort_values("median_d", ascending=False).to_string(index=False, float_format="%.3f"))

    # Soramic-drift vs Lausanne-drift side-by-side
    pivot = df[df["comparison"].isin(["resection_vs_soramic", "resection_vs_lausanne"])].pivot(
        index=["model_id", "input"], columns="comparison", values=["median_d", "frac_sig"]
    )
    pivot.columns = ["_".join(c[::-1]).replace("resection_vs_", "") for c in pivot.columns]
    pivot = pivot.rename(columns={
        "soramic_median_d": "soramic_d",
        "lausanne_median_d": "lausanne_d",
        "soramic_frac_sig": "soramic_sig",
        "lausanne_frac_sig": "lausanne_sig",
    })
    pivot["Δ(lau−sor)"] = pivot["lausanne_d"] - pivot["soramic_d"]
    pivot = pivot.sort_values("Δ(lau−sor)", ascending=False).reset_index()
    print(f"\n{'='*60}")
    print("  Soramic vs Lausanne drift from resection  (sorted by Δ)")
    print(f"{'='*60}")
    print(pivot.to_string(index=False, float_format="%.3f"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "results" / "eval" / "embedding_drift.csv",
    )
    args = parser.parse_args()
    _print_summary(run(args.out))


if __name__ == "__main__":
    main()
