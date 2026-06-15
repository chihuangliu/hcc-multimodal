"""Arterial radiomic baseline for RFS prediction with SelectKBest feature selection."""

import argparse
import matplotlib
matplotlib.use('Agg')

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.feature_selection import SelectKBest, f_classif
from matplotlib_venn import venn3

from hcc_multimodal.baselines.config import MODELS_LR, MODELS_RF
from hcc_multimodal.baselines.data import add_rfs_columns
from hcc_multimodal.baselines.transforms import DataType, CLINICAL_X_COLUMNS
from hcc_multimodal.baselines.evaluation import (
    run_cv_experiment,
    plot_cv_results,
    apply_selector_before_cv,
)

ROOT = Path(__file__).resolve().parent.parent.parent


def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir) if args.output_dir else ROOT / "results" / "baseline" / "radiomic_arterial_rfs"
    output_dir.mkdir(parents=True, exist_ok=True)

    clinical_data = pd.read_csv(
        ROOT / "data" / "Clinical" / "2025_Nov_18_ICL_Resection_Clinical_Outcome_soramic_format.csv"
    ).dropna(how="all")
    arterial_data = pd.read_csv(
        ROOT / "data" / "Radiomics" / "arterial_radiomics.csv"
    ).dropna(how="all")

    arterial_data["SID"] = (
        arterial_data["Scan name"].str.replace(r"\.nii\.gz$", "", regex=True).astype(int)
    )
    arterial_data = arterial_data.drop(columns=["Scan name", "VOI name", "Scan path", "VOI path"])
    feature_cols = [c for c in arterial_data.columns if c != "SID"]
    print(f"Arterial radiomic features: {len(feature_cols)}, patients: {len(arterial_data)}")

    clinical_data = add_rfs_columns(clinical_data)

    arterial_rfs = arterial_data.merge(
        clinical_data[["SID", "rfs_1year", "rfs_2year"]], on="SID"
    )
    arterial_rfs = arterial_rfs.set_index("SID").sort_index()
    print(f"Matched patients: {len(arterial_rfs)}")

    confounding_vars = args.confounding_vars if args.confounding_vars else None
    if confounding_vars:
        confound_df = clinical_data.set_index("SID")[confounding_vars].copy()
        confound_df = confound_df.loc[confound_df.index.isin(arterial_rfs.index)]
        confound_df["Sex"] = confound_df["Sex"].map({1.0: "male", 2.0: "female"})
        confound_x_cols = {v: CLINICAL_X_COLUMNS[v] for v in confounding_vars}
        print(f"Confounding variables: {confounding_vars}")
    else:
        confound_df = None
        confound_x_cols = None

    X = arterial_rfs.drop(columns=["rfs_1year", "rfs_2year"])
    x_columns = {col: DataType.CONTINUOUS for col in X}
    selector = SelectKBest(f_classif, k=args.select_k)

    all_records = []
    all_fold_records = []

    for rfs_year in args.rfs_years:
        y = arterial_rfs[f"rfs_{rfs_year}year"]
        experiment_label = f"RFS {rfs_year} year - Arterial Radiomics"

        X_cv, y_cv, x_columns_cv, cv_kwargs = apply_selector_before_cv(
            X, y, x_columns, selector,
            selector_first=False,
            label=experiment_label,
            save_path=output_dir / f"rfs_{rfs_year}y_preselected_features.csv",
        )

        records_lr, fold_records_lr = run_cv_experiment(
            X_cv, y_cv, x_columns_cv, experiment_label,
            models=MODELS_LR, n_splits=args.cv_folds,
            X_confound=confound_df, confound_x_columns=confound_x_cols,
            **cv_kwargs,
        )
        plot_cv_results(
            fold_records_lr,
            title=f"{experiment_label}, Logistic Regression",
            save_path=output_dir / f"rfs_{rfs_year}y_lr.png",
        )

        records_rf, fold_records_rf = run_cv_experiment(
            X_cv, y_cv, x_columns_cv, experiment_label,
            models=MODELS_RF, n_splits=args.cv_folds,
            X_confound=confound_df, confound_x_columns=confound_x_cols,
            **cv_kwargs,
        )
        plot_cv_results(
            fold_records_rf,
            title=f"{experiment_label}, Random Forest",
            save_path=output_dir / f"rfs_{rfs_year}y_rf.png",
        )

        all_records.extend([records_lr, records_rf])
        all_fold_records.append((rfs_year, fold_records_lr))

        # Save non-zero L1 LR features
        rows = []
        for _, row in fold_records_lr.iterrows():
            if row["lr_nonzero_features"] is not None and len(row["lr_nonzero_features"]) > 0:
                df = row["lr_nonzero_features"].copy()
                df["model"] = row["model"]
                df["fold"] = row["fold"]
                rows.append(df)

        if rows:
            combined = pd.concat(rows, ignore_index=True)
            combined.to_csv(
                output_dir / f"nonzero_features_rfs_{rfs_year}y_all_folds.csv", index=False
            )
            union_features = sorted(combined["feature"].unique())
            txt_path = output_dir / f"nonzero_features_rfs_{rfs_year}y_union.txt"
            with open(txt_path, "w") as f:
                f.write(f"# Non-zero L1-regularised LR features for RFS {rfs_year}-year\n")
                f.write(f"# Union across all folds and C values\n")
                f.write(f"# Total: {len(union_features)} unique features\n\n")
                for feat in union_features:
                    f.write(feat + "\n")
            print(f"RFS {rfs_year}y: {len(union_features)} unique non-zero features saved to {txt_path}")
        else:
            print(f"RFS {rfs_year}y: no non-zero features found")

    # Save summary and fold records
    summary = pd.concat(all_records, ignore_index=True)
    summary.to_csv(output_dir / "summary.csv", index=False)

    fold_records_to_save = []
    for rfs_year, fold_rec_lr in all_fold_records:
        tmp = fold_rec_lr.drop(columns=["selected_features", "lr_nonzero_features"], errors="ignore").copy()
        tmp["rfs_year"] = rfs_year
        tmp["model_type"] = "lr"
        fold_records_to_save.append(tmp)
    pd.concat(fold_records_to_save, ignore_index=True).to_csv(output_dir / "fold_records.csv", index=False)

    print("\nSummary:")
    print(summary.to_string(index=False))

    # Venn diagrams for LR C=1 non-zero features across CV folds
    if len(all_fold_records) > 0:
        n_years = len(all_fold_records)
        fig, axes = plt.subplots(1, n_years, figsize=(6 * n_years, 5))
        if n_years == 1:
            axes = [axes]

        for ax, (rfs_year, fold_records_lr) in zip(axes, all_fold_records):
            c1_rows = fold_records_lr[
                (fold_records_lr["model"] == "LR_C=1")
                & fold_records_lr["lr_nonzero_features"].apply(
                    lambda x: x is not None and len(x) > 0
                )
            ]
            fold_sets = {}
            for _, row in c1_rows.iterrows():
                fold_sets[row["fold"]] = set(row["lr_nonzero_features"]["feature"])

            sets = [fold_sets.get(i, set()) for i in [1, 2, 3]]
            venn3(sets, set_labels=("Fold 1", "Fold 2", "Fold 3"), ax=ax)
            ax.set_title(f"RFS {rfs_year}-year LR C=1 — non-zero features", fontsize=12)

        plt.suptitle("Non-zero L1 LR (C=1) features per CV fold", fontsize=13)
        plt.tight_layout()
        plt.savefig(output_dir / "rfs_lr_c1_venn.png", bbox_inches="tight", dpi=150)
        plt.close('all')

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Arterial radiomic baseline for RFS prediction."
    )
    parser.add_argument("--rfs_years", nargs="+", type=int, default=[1, 2],
                        help="RFS year thresholds to evaluate (default: 1 2)")
    parser.add_argument("--cv_folds", type=int, default=3,
                        help="Number of cross-validation folds (default: 3)")
    parser.add_argument("--select_k", type=int, default=100,
                        help="SelectKBest k for feature selection (default: 100)")
    parser.add_argument("--confounding_vars", nargs="*", default=["Age", "Sex"],
                        help="Confounding variables to include (default: Age Sex)")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (default: results/baseline/radiomic_arterial_rfs)")
    main(parser.parse_args())
