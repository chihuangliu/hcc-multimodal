"""RNA-seq baseline for RFS prediction with DESeq2 or SelectKBest feature selection."""

import argparse
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.feature_selection import SelectKBest, f_classif

from hcc_multimodal.baselines.config import MODELS_LR, MODELS_RF
from hcc_multimodal.baselines.data import add_rfs_columns, get_hcc_genes
from hcc_multimodal.baselines.transforms import DataType, CLINICAL_X_COLUMNS
from hcc_multimodal.baselines.evaluation import (
    run_cv_experiment,
    plot_cv_results,
    DeseqCPMSelector,
    apply_selector_before_cv,
)
from hcc_multimodal.utils.data import CLINICAL_CSV, RNA_SEQ_CSV

ROOT = Path(__file__).resolve().parent.parent.parent


def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir) if args.output_dir else ROOT / "results" / "baseline" / "rna_rfs"
    output_dir.mkdir(parents=True, exist_ok=True)

    clinical_data = pd.read_csv(CLINICAL_CSV).dropna(how="all")

    rna_raw = pd.read_csv(RNA_SEQ_CSV).dropna(how="all")

    # Transpose to (samples × genes)
    columns = rna_raw["Gene Symbol"]
    rna_data = rna_raw.T.iloc[2:, :]
    rna_data.columns = columns.values
    rna_data = rna_data.loc[:, ~pd.isnull(rna_data.columns)]
    rna_data = rna_data.loc[:, ~rna_data.columns.duplicated()]
    rna_data = rna_data.apply(pd.to_numeric, errors="coerce")
    rna_data.index = rna_data.index.astype(int)
    rna_data = rna_data.rename_axis("SID").reset_index()
    print(f"RNA data shape (patients × genes): {rna_data.shape}")

    clinical_data = add_rfs_columns(clinical_data)
    print(clinical_data["rfs_1year"].value_counts(dropna=False).to_string())
    print(clinical_data["rfs_2year"].value_counts(dropna=False).to_string())

    rna_count = rna_data.drop(columns=["SID"])
    rna_count.index = rna_data["SID"]

    if args.predefined_genes:
        hcc_genes = get_hcc_genes()
        rna_count = rna_count.loc[:, rna_count.columns.intersection(hcc_genes)]
        print(f"After predefined gene filtering: {rna_count.shape[1]}")

    rna_rfs = rna_count.merge(
        clinical_data[["SID", "rfs_1year", "rfs_2year"]], on="SID"
    )
    rna_rfs = rna_rfs.set_index("SID").sort_index()

    selector = DeseqCPMSelector(pvalue=args.deseq_pvalue, min_features=args.deseq_min_features)
    selector_first = True
    rna_cpm_flag = False

    confounding_vars = args.confounding_vars if args.confounding_vars else None
    if confounding_vars:
        confound_df = clinical_data.set_index("SID")[confounding_vars].copy()
        confound_df = confound_df.loc[confound_df.index.isin(rna_rfs.index)]
        confound_df["Sex"] = confound_df["Sex"].map({1.0: "male", 2.0: "female"})
        confound_x_cols = {v: CLINICAL_X_COLUMNS[v] for v in confounding_vars}
        print(f"Confounding variables: {confounding_vars}")
    else:
        confound_df = None
        confound_x_cols = None

    X = rna_rfs.drop(columns=["rfs_1year", "rfs_2year"])
    x_columns = {col: DataType.CONTINUOUS for col in X}

    np.random.seed(42)
    all_records = []
    all_fold_records = []

    for rfs_year in args.rfs_years:
        y = rna_rfs[f"rfs_{rfs_year}year"]
        experiment_label = f"RFS {rfs_year} year - RNA-seq"

        X_cv, y_cv, x_columns_cv, cv_kwargs = apply_selector_before_cv(
            X, y, x_columns, selector,
            selector_first=selector_first, rna_cpm=rna_cpm_flag,
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
            title=f"{experiment_label}, Logistic Regression (DESeq2)",
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
            title=f"{experiment_label}, Random Forest (DESeq2)",
            save_path=output_dir / f"rfs_{rfs_year}y_rf.png",
        )

        # Save selected features and non-zero LR coefficients
        for fold_records, model_type in [(fold_records_lr, "lr"), (fold_records_rf, "rf")]:
            feats = []
            for _, row in fold_records.iterrows():
                if row["selected_features"] is not None:
                    df = row["selected_features"].copy()
                    df["fold"] = row["fold"]
                    df["model"] = row["model"]
                    feats.append(df)
            if feats:
                pd.concat(feats, ignore_index=True).to_csv(
                    output_dir / f"rfs_{rfs_year}y_{model_type}_selected_features.csv", index=False
                )

            nonzero = []
            for _, row in fold_records.iterrows():
                if row["lr_nonzero_features"] is not None:
                    df = row["lr_nonzero_features"].copy()
                    df["fold"] = row["fold"]
                    df["model"] = row["model"]
                    nonzero.append(df)
            if nonzero:
                pd.concat(nonzero, ignore_index=True).to_csv(
                    output_dir / f"rfs_{rfs_year}y_{model_type}_lr_nonzero_features.csv", index=False
                )

            fold_records_to_save = fold_records.drop(
                columns=["selected_features", "lr_nonzero_features"], errors="ignore"
            ).copy()
            fold_records_to_save["rfs_year"] = rfs_year
            fold_records_to_save["model_type"] = model_type
            all_fold_records.append(fold_records_to_save)

        all_records.extend([records_lr, records_rf])

    summary = pd.concat(all_records, ignore_index=True)
    summary.to_csv(output_dir / "summary.csv", index=False)
    pd.concat(all_fold_records, ignore_index=True).to_csv(output_dir / "fold_records.csv", index=False)

    print("\nSummary:")
    print(summary.to_string(index=False))
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RNA-seq baseline for RFS prediction with DESeq2 feature selection."
    )
    parser.add_argument("--rfs_years", nargs="+", type=int, default=[1, 2],
                        help="RFS year thresholds to evaluate (default: 1 2)")
    parser.add_argument("--cv_folds", type=int, default=3,
                        help="Number of cross-validation folds (default: 3)")
    parser.add_argument("--deseq_pvalue", type=float, default=0.05,
                        help="DESeq2 adjusted p-value threshold (default: 0.05)")
    parser.add_argument("--deseq_min_features", type=int, default=20,
                        help="Minimum features to keep if DESeq2 finds too few (default: 20)")
    parser.add_argument("--predefined_genes", action="store_true",
                        help="Pre-filter to HCC gene list before DESeq2 (default: off)")
    parser.add_argument("--confounding_vars", nargs="*", default=["Age", "Sex"],
                        help="Confounding variables to include (default: Age Sex)")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (default: results/baseline/rna_rfs)")
    main(parser.parse_args())
