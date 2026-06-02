"""Regenerate CV strip plots (SVG + PNG) from a saved fold_records.csv.

No cross-validation is re-run — the fold_records.csv produced by the
rna_baseline_rfs notebook is sufficient.

Usage:
    # Re-plot a single directory
    python scripts/replot_cv_results.py reports/0504/deseq_p0.05_before_cv_all_genes

    # Re-plot all 0504 RNA experiment directories at once
    python scripts/replot_cv_results.py reports/0504/deseq_* reports/0504/kbest_*
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # non-interactive backend — no display windows
import pandas as pd

matplotlib.rcParams['svg.fonttype'] = 'none'

from hcc_multimodal.baselines.evaluation import plot_cv_results


def replot_dir(output_dir: Path) -> None:
    fold_csv = output_dir / "fold_records.csv"
    if not fold_csv.exists():
        print(f"  SKIP {output_dir} — no fold_records.csv")
        return

    fold_df = pd.read_csv(fold_csv)

    rfs_years = fold_df["rfs_year"].unique() if "rfs_year" in fold_df.columns else [None]
    model_types = fold_df["model_type"].unique() if "model_type" in fold_df.columns else [None]

    for rfs_year in sorted(rfs_years):
        for model_type in sorted(model_types):
            mask = pd.Series(True, index=fold_df.index)
            if rfs_year is not None:
                mask &= fold_df["rfs_year"] == rfs_year
            if model_type is not None:
                mask &= fold_df["model_type"] == model_type
            subset = fold_df[mask]
            if subset.empty:
                continue

            experiment = subset["experiment"].iloc[0]
            model_label = "Logistic Regression" if model_type == "lr" else "Random Forest"
            title = f"{experiment}, {model_label}"

            yr_str = f"{rfs_year}y" if rfs_year is not None else "rfs"
            mt_str = model_type or "model"
            save_path = output_dir / f"rfs_{yr_str}_{mt_str}.png"

            plot_cv_results(subset, title=title, save_path=save_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dirs", nargs="+", type=Path, help="Output directories containing fold_records.csv")
    args = parser.parse_args()

    for d in args.dirs:
        print(f"Processing {d} ...")
        replot_dir(d)


if __name__ == "__main__":
    main()
