"""Ensemble RFS prediction combining arterial radiomics and RNA-seq via probability averaging."""

import argparse
import warnings
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score

from hcc_multimodal.baselines.config import MODELS_LR, MODELS_RF
from hcc_multimodal.baselines.data import add_rfs_columns
from hcc_multimodal.baselines.transforms import DataType, CLINICAL_X_COLUMNS
from hcc_multimodal.baselines.evaluation import (
    run_cv_experiment,
    DeseqCPMSelector,
    apply_selector_before_cv,
)

ROOT = Path(__file__).resolve().parent.parent.parent

warnings.filterwarnings("ignore")


def main(args: argparse.Namespace) -> None:
    rfs_year = args.rfs_year
    confounding_vars = args.confounding_vars if args.confounding_vars else None
    _conf_suffix = ("_" + "_".join(v.lower() for v in confounding_vars)) if confounding_vars else ""
    output_dir = (
        Path(args.output_dir) if args.output_dir
        else ROOT / "results" / "baseline" / f"ensemble_{rfs_year}y_lr{_conf_suffix}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    clinical_data = pd.read_csv(
        ROOT / "data" / "Clinical" / "2025_Nov_18_ICL_Resection_Clinical_Outcome_soramic_format.csv"
    ).dropna(how="all")
    clinical_data = add_rfs_columns(clinical_data)

    arterial_data = pd.read_csv(ROOT / "data" / "Radiomics" / "arterial_radiomics.csv").dropna(how="all")
    arterial_data["SID"] = (
        arterial_data["Scan name"].str.replace(r"\.nii\.gz$", "", regex=True).astype(int)
    )
    arterial_data = arterial_data.drop(columns=["Scan name", "VOI name", "Scan path", "VOI path"])

    rna_raw = pd.read_csv(ROOT / "data" / "RNA_seq" / "Matrix_output_radiology_only.csv").dropna(how="all")
    gene_symbols = rna_raw["Gene Symbol"]
    rna_raw = rna_raw.T.iloc[2:, :]
    rna_raw.columns = gene_symbols.values
    rna_raw = rna_raw.loc[:, ~pd.isnull(rna_raw.columns)]
    rna_raw = rna_raw.loc[:, ~rna_raw.columns.duplicated()]
    rna_raw = rna_raw.apply(pd.to_numeric, errors="coerce")
    rna_raw.index = rna_raw.index.astype(int)
    rna_raw = rna_raw.rename_axis("SID").reset_index()
    print(f"Arterial features: {arterial_data.shape[1] - 1}, patients: {len(arterial_data)}")
    print(f"RNA-seq genes: {rna_raw.shape[1] - 1}, patients: {len(rna_raw)}")

    y_col = f"rfs_{rfs_year}year"
    arterial_rfs = arterial_data.merge(clinical_data[["SID", y_col]], on="SID").set_index("SID").sort_index()
    rna_rfs = rna_raw.merge(clinical_data[["SID", y_col]], on="SID").set_index("SID").sort_index()

    common_sids = arterial_rfs.index.intersection(rna_rfs.index)
    arterial_rfs = arterial_rfs.loc[common_sids]
    rna_rfs = rna_rfs.loc[common_sids]
    y_common = rna_rfs[y_col].dropna()
    common_sids = y_common.index
    arterial_rfs = arterial_rfs.loc[common_sids]
    rna_rfs = rna_rfs.loc[common_sids]
    print(f"{rfs_year}y RFS — common labeled patients: {len(common_sids)}, positives: {int(y_common.sum())}")

    if confounding_vars:
        confound_df = clinical_data.set_index("SID")[confounding_vars].copy()
        confound_df = confound_df.loc[confound_df.index.isin(common_sids)]
        confound_df["Sex"] = confound_df["Sex"].map({1.0: "male", 2.0: "female"})
        confound_x_cols = {v: CLINICAL_X_COLUMNS[v] for v in confounding_vars}
        print(f"Confounding variables: {confounding_vars}")
    else:
        confound_df = None
        confound_x_cols = None

    np.random.seed(42)

    # Feature selection before CV
    X_rad = arterial_rfs.drop(columns=[y_col])
    x_cols_rad = {col: DataType.CONTINUOUS for col in X_rad}
    X_cv_rad, y_cv, x_cols_cv_rad, cv_kwargs_rad = apply_selector_before_cv(
        X_rad, y_common, x_cols_rad, SelectKBest(f_classif, k=100),
        selector_first=False, label=f"Radiomics {rfs_year}y",
    )

    X_rna = rna_rfs.drop(columns=[y_col])
    x_cols_rna = {col: DataType.CONTINUOUS for col in X_rna}
    X_cv_rna, _, x_cols_cv_rna, cv_kwargs_rna = apply_selector_before_cv(
        X_rna, y_common, x_cols_rna, DeseqCPMSelector(pvalue=0.05, min_features=20),
        selector_first=True, label=f"RNA-seq {rfs_year}y",
    )

    # Best models: RF for radiomics, LR for RNA-seq
    rad_model_key = "RF_max_depth=2_min_samples_leaf=10"
    rna_model_key = "LR_C=1"
    model_rad = {rad_model_key: MODELS_RF[rad_model_key]}
    model_rna = {rna_model_key: MODELS_LR[rna_model_key]}

    _, fold_df_rad = run_cv_experiment(
        X_cv_rad, y_cv, x_cols_cv_rad, f"Radiomics {rfs_year}y",
        models=model_rad, n_splits=args.cv_folds, return_proba=True,
        X_confound=confound_df, confound_x_columns=confound_x_cols,
        **cv_kwargs_rad,
    )
    _, fold_df_rna = run_cv_experiment(
        X_cv_rna, y_cv, x_cols_cv_rna, f"RNA-seq {rfs_year}y",
        models=model_rna, n_splits=args.cv_folds, return_proba=True,
        X_confound=confound_df, confound_x_columns=confound_x_cols,
        **cv_kwargs_rna,
    )

    # Ensemble by averaging probabilities
    rows_rad = fold_df_rad[fold_df_rad["model"] == rad_model_key].set_index("fold")
    rows_rna = fold_df_rna[fold_df_rna["model"] == rna_model_key].set_index("fold")

    fold_aucs_ens, fold_aucs_rad, fold_aucs_rna = [], [], []
    for fold_i in range(1, args.cv_folds + 1):
        r_rad = rows_rad.loc[fold_i]
        r_rna = rows_rna.loc[fold_i]
        assert r_rad["test_indices"] == r_rna["test_indices"], f"Fold {fold_i} indices mismatch"
        ens_proba = (np.array(r_rad["test_proba"]) + np.array(r_rna["test_proba"])) / 2
        y_test = y_cv.loc[r_rad["test_indices"]]
        auc_ens = roc_auc_score(y_test, ens_proba)
        fold_aucs_ens.append(auc_ens)
        fold_aucs_rad.append(r_rad["test_auc"])
        fold_aucs_rna.append(r_rna["test_auc"])
        print(f"Fold {fold_i}: ensemble={auc_ens:.3f}  radiomics={r_rad['test_auc']:.3f}  rna={r_rna['test_auc']:.3f}")

    print(f"\nMean AUC — ensemble={np.mean(fold_aucs_ens):.3f}  radiomics={np.mean(fold_aucs_rad):.3f}  rna={np.mean(fold_aucs_rna):.3f}")

    results = pd.DataFrame({
        "fold": list(range(1, args.cv_folds + 1)) + ["mean"],
        "ensemble": fold_aucs_ens + [np.mean(fold_aucs_ens)],
        "radiomics": fold_aucs_rad + [np.mean(fold_aucs_rad)],
        "rna_seq": fold_aucs_rna + [np.mean(fold_aucs_rna)],
    })
    results.to_csv(output_dir / "auc.csv", index=False)

    # Plot
    matplotlib.rcParams['svg.fonttype'] = 'none'
    conf_label = " + " + "/".join(confounding_vars) if confounding_vars else ""
    fig, ax = plt.subplots(figsize=(7, 5))
    folds = range(1, args.cv_folds + 1)
    ax.plot(folds, fold_aucs_rad, "o--", color="steelblue", label=f"Radiomics (mean={np.mean(fold_aucs_rad):.3f})")
    ax.plot(folds, fold_aucs_rna, "s--", color="seagreen", label=f"RNA-seq (mean={np.mean(fold_aucs_rna):.3f})")
    ax.plot(folds, fold_aucs_ens, "D-", color="tomato", linewidth=2, label=f"Ensemble (mean={np.mean(fold_aucs_ens):.3f})")
    ax.axhline(np.mean(fold_aucs_ens), color="tomato", linestyle=":", linewidth=1, alpha=0.6)
    ax.set_xlabel("Cross-validation Fold", fontsize=16)
    ax.set_ylabel("Area Under the ROC Curve (AUC)", fontsize=16)
    ax.set_title(f"{rfs_year}-year RFS — Ensemble{conf_label} (before-CV)", fontsize=16)
    ax.set_xticks(list(folds))
    ax.tick_params(labelsize=14)
    ax.legend(fontsize=14)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "auc.png", bbox_inches="tight", dpi=150)
    plt.savefig(output_dir / "auc.svg", bbox_inches="tight")
    plt.close(fig)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ensemble RFS prediction: radiomics + RNA-seq probability averaging."
    )
    parser.add_argument("--rfs_year", type=int, default=2,
                        help="RFS year threshold (default: 2)")
    parser.add_argument("--cv_folds", type=int, default=3,
                        help="Number of cross-validation folds (default: 3)")
    parser.add_argument("--confounding_vars", nargs="*", default=["Age", "Sex"],
                        help="Confounding variables to include (default: Age Sex)")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (default: results/baseline/ensemble_Ny_lr[_confounders])")
    main(parser.parse_args())
