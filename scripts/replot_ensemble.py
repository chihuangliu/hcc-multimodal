"""Regenerate the ensemble AUC line plot (SVG + PNG) from a saved auc.csv.

No cross-validation is re-run — the auc.csv produced by the
ensemble_baseline_rfs notebook is sufficient.

Usage:
    python scripts/replot_ensemble.py reports/0504/ensemble_2y_lr_age_sex
    python scripts/replot_ensemble.py reports/0504/ensemble_2y_lr_age_sex reports/0504/ensemble_2y_lr
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams['svg.fonttype'] = 'none'


def replot_dir(output_dir: Path) -> None:
    auc_csv = output_dir / "auc.csv"
    if not auc_csv.exists():
        print(f"  SKIP {output_dir} — no auc.csv")
        return

    df = pd.read_csv(auc_csv)
    df_folds = df[df["fold"].apply(lambda x: str(x).isdigit())].copy()
    df_folds["fold"] = df_folds["fold"].astype(int)

    folds = df_folds["fold"].tolist()
    fold_aucs_ens = df_folds["ensemble"].tolist()
    fold_aucs_rad = df_folds["radiomics"].tolist()
    fold_aucs_rna = df_folds["rna_seq"].tolist()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(folds, fold_aucs_rad, "o--", color="steelblue",
            label=f"Radiomics (mean={np.mean(fold_aucs_rad):.3f})")
    ax.plot(folds, fold_aucs_rna, "s--", color="seagreen",
            label=f"RNA-seq (mean={np.mean(fold_aucs_rna):.3f})")
    ax.plot(folds, fold_aucs_ens, "D-", color="tomato", linewidth=2,
            label=f"Ensemble (mean={np.mean(fold_aucs_ens):.3f})")
    ax.axhline(np.mean(fold_aucs_ens), color="tomato", linestyle=":", linewidth=1, alpha=0.6)
    ax.set_xlabel("Cross-validation Fold", fontsize=16)
    ax.set_ylabel("Area Under the ROC Curve (AUC)", fontsize=16)
    ax.set_title(f"Ensemble — {output_dir.name}", fontsize=16)
    ax.set_xticks(folds)
    ax.tick_params(labelsize=14)
    ax.legend(fontsize=14)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    fig.savefig(output_dir / "auc.png", bbox_inches="tight", dpi=150)
    fig.savefig(output_dir / "auc.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_dir}/auc.png (+ .svg)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dirs", nargs="+", type=Path, help="Ensemble output directories containing auc.csv")
    args = parser.parse_args()

    for d in args.dirs:
        replot_dir(d)


if __name__ == "__main__":
    main()
