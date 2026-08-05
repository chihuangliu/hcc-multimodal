"""KS-test distribution drift analysis for contrastive embeddings.

Compares patient-level embedding distributions across three cohorts:
  resection (training) · soramic (ablation test) · lausanne (external test)

For each model and each cohort pair, runs a per-dimension KS test and reports:
  median_d     – median KS D-statistic across embedding dimensions
  mean_d       – mean   KS D-statistic across embedding dimensions
  frac_sig     – fraction of dimensions with raw p < 0.05
  frac_sig_bh  – fraction of dimensions with Benjamini--Hochberg q < 0.05

Usage:
  python -m hcc_multimodal.eval.embedding_drift [--out PATH]
  python -m hcc_multimodal.eval.embedding_drift --model-id d7085bf5 --input raw
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from hcc_multimodal.eval.data import (
    TRAINING_ROOT,
    load_ablation_outcomes,
    load_resection_outcomes,
)
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


def bh_qvalues(p_vals: np.ndarray) -> np.ndarray:
    """Benjamini--Hochberg adjusted p-values."""
    n = p_vals.size
    order = np.argsort(p_vals)
    ranked = p_vals[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.clip(ranked, 0.0, 1.0)
    return q


def ks_drift(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    n_dims = a.shape[1]
    d_stats = np.empty(n_dims)
    p_vals  = np.empty(n_dims)
    for i in range(n_dims):
        res = ks_2samp(a[:, i], b[:, i])
        d_stats[i] = res.statistic
        p_vals[i]  = res.pvalue
    return {
        "n_dims":      n_dims,
        "median_d":    float(np.median(d_stats)),
        "mean_d":      float(np.mean(d_stats)),
        "max_d":       float(np.max(d_stats)),
        "frac_sig":    float(np.mean(p_vals < 0.05)),
        "frac_sig_bh": float(np.mean(bh_qvalues(p_vals) < 0.05)),
    }


def _outcomes(cohort: str, target: str) -> pd.Series:
    """``target`` labels, indexed by SID — the patients the downstream head is
    actually fitted or evaluated on."""
    if cohort == "resection":
        return load_resection_outcomes(target)
    return load_ablation_outcomes(_COHORT_FILENAME[cohort], target)


def run(
    out_path: Path,
    configs: list[tuple[str, str]] = MODEL_CONFIGS,
    target: str | None = None,
    stratify: bool = False,
) -> pd.DataFrame:
    if stratify and target is None:
        raise ValueError("stratify=True requires a target")

    records = []
    for model_id, suffix in configs:
        embs: dict[str, pd.DataFrame] = {}
        labels: dict[str, pd.Series] = {}
        for cohort in ("resection", "soramic", "lausanne"):
            df = pd.read_parquet(_emb_path(model_id, suffix, cohort))
            if target is not None:
                y = _outcomes(cohort, target)
                idx = df.index.intersection(y.index)
                df, labels[cohort] = df.loc[idx], y.loc[idx]
            embs[cohort] = df

        # (stratum label, row mask per cohort); "all" = no label stratification
        strata: list[tuple[str, int | None]] = [("all", None)]
        if stratify:
            strata += [("positives", 1), ("negatives", 0)]

        for cohort_a, cohort_b in COMPARISONS:
            for stratum, value in strata:
                a, b = embs[cohort_a], embs[cohort_b]
                if value is not None:
                    a = a[labels[cohort_a] == value]
                    b = b[labels[cohort_b] == value]
                records.append(
                    {
                        "model_id":   model_id,
                        "input":      suffix,
                        "comparison": f"{cohort_a}_vs_{cohort_b}",
                        "stratum":    stratum,
                        "n_a":        a.shape[0],
                        "n_b":        b.shape[0],
                        **ks_drift(a.values, b.values),
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
        cols = ["model_id", "input", "stratum", "n_a", "n_b", "median_d", "mean_d",
                "max_d", "frac_sig", "frac_sig_bh"]
        show = grp[cols].copy()
        print(show.sort_values(["stratum", "median_d"], ascending=[True, False])
                  .to_string(index=False, float_format="%.3f"))

    # Soramic-drift vs Lausanne-drift side-by-side (unstratified rows only)
    unstratified = df[df["stratum"] == "all"]
    pivot = unstratified[
        unstratified["comparison"].isin(["resection_vs_soramic", "resection_vs_lausanne"])
    ].pivot(
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
    parser.add_argument(
        "--model-id",
        nargs="+",
        help="Restrict the run to these model ids (default: the full MODEL_CONFIGS list).",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        choices=["raw", "bbox"],
        help="Input suffix per --model-id; a single value applies to all. "
             "Defaults to the suffix recorded in MODEL_CONFIGS.",
    )
    parser.add_argument(
        "--target",
        help="Restrict every cohort to patients with a non-missing label for this "
             "outcome (e.g. rfs_2year). Default: all cached patients.",
    )
    parser.add_argument(
        "--stratify-by-label",
        action="store_true",
        help="In addition to the pooled comparison, repeat each cohort pair within the "
             "positive and negative strata of --target.",
    )
    args = parser.parse_args()

    configs = MODEL_CONFIGS
    if args.model_id:
        known = dict(MODEL_CONFIGS)
        if args.input and len(args.input) == 1:
            suffixes = args.input * len(args.model_id)
        elif args.input:
            if len(args.input) != len(args.model_id):
                parser.error("--input must have 1 value or one per --model-id")
            suffixes = args.input
        else:
            missing = [m for m in args.model_id if m not in known]
            if missing:
                parser.error(f"--input required for unlisted model ids: {missing}")
            suffixes = [known[m] for m in args.model_id]
        configs = list(zip(args.model_id, suffixes))

    if args.stratify_by_label and not args.target:
        parser.error("--stratify-by-label requires --target")

    _print_summary(
        run(args.out, configs, target=args.target, stratify=args.stratify_by_label)
    )


if __name__ == "__main__":
    main()
