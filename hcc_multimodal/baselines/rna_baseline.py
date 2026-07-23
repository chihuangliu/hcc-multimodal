"""RNA-seq baseline: raw counts (v1) and filtered/normalised experiments (v2, merged).

Experiment sets
---------------
basic   — RNA→death, RNA→OS, RNA+Clinical→death, RNA+Clinical→OS; with/without grid search
pipeline — on filtered log2(CPM) RNA: SelectKBest+PCA (Exp A) and SelectKBest+L1 (Exp B)
"""

import argparse
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from hcc_multimodal.baselines.config import PARAM_GRIDS, RANDOM_STATE
from hcc_multimodal.baselines.evaluation import (
    run_cv_experiment,
    plot_pca_variance,
    plot_cv_results,
    plot_experiment_comparison,
)
from hcc_multimodal.baselines.transforms import CLINICAL_X_COLUMNS, DataType, build_preprocessor
from hcc_multimodal.utils.data import CLINICAL_CSV, RADIOMIC_CLUSTER_CSV, RNA_SEQ_CSV

ROOT = Path(__file__).resolve().parent.parent.parent


def _run_pipeline_experiment(
    X: pd.DataFrame,
    y: pd.Series,
    x_columns: dict,
    label: str,
    models: dict,
    select_k: int | None = None,
    use_pca: bool = True,
    n_splits: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run stratified k-fold CV with a configurable sklearn pipeline."""
    preprocessor = build_preprocessor(x_columns)
    outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    mask = ~y.isna()
    X_clean, y_clean = X[mask], y[mask]
    print(f"[{label}]  n={len(y_clean)}  positives={int(y_clean.sum())}  features={X_clean.shape[1]}")

    records, fold_records = [], []
    for model_name, model in models.items():
        steps = [("preprocessor", preprocessor)]
        if select_k is not None:
            steps.append(("select_k", SelectKBest(f_classif, k=select_k)))
        if use_pca:
            steps.append(("pca", PCA(n_components=0.9, random_state=RANDOM_STATE)))
        steps.append(("model", model))
        pipe = Pipeline(steps)

        scores = cross_validate(
            pipe, X_clean, y_clean, cv=outer_cv,
            scoring=["roc_auc", "accuracy"], return_train_score=True,
        )
        records.append({
            "experiment": label, "model": model_name,
            "AUC mean": round(scores["test_roc_auc"].mean(), 3),
            "AUC std": round(scores["test_roc_auc"].std(), 3),
            "Accuracy mean": round(scores["test_accuracy"].mean(), 3),
            "Accuracy std": round(scores["test_accuracy"].std(), 3),
        })
        for fold_i, (train_auc, test_auc) in enumerate(
            zip(scores["train_roc_auc"], scores["test_roc_auc"])
        ):
            fold_records.append({
                "experiment": label, "model": model_name,
                "fold": fold_i + 1, "train_auc": train_auc,
                "test_auc": test_auc,
            })
    return pd.DataFrame(records), pd.DataFrame(fold_records)


def _load_rna_raw() -> pd.DataFrame:
    """Load and transpose RNA-seq count matrix to (samples × genes)."""
    rna_raw = pd.read_csv(RNA_SEQ_CSV).dropna(how="all")
    columns = rna_raw["Gene Symbol"]
    rna = rna_raw.T.iloc[2:, :]
    rna.columns = columns.values
    rna = rna.loc[:, ~pd.isnull(rna.columns)]
    rna = rna.loc[:, ~rna.columns.duplicated()]
    rna = rna.apply(pd.to_numeric, errors="coerce")
    rna.index = rna.index.astype(int)
    rna = rna.rename_axis("SID").reset_index()
    print(f"RNA data shape (patients × genes): {rna.shape}")
    return rna


def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir) if args.output_dir else ROOT / "results" / "baseline" / "rna_baseline"
    output_dir.mkdir(parents=True, exist_ok=True)

    clinical_data = pd.read_csv(CLINICAL_CSV).dropna(how="all")
    radiomics_data = pd.read_csv(RADIOMIC_CLUSTER_CSV).dropna(how="all")
    radiomic_death = radiomics_data[["SID", "death"]].copy()

    # ── Load raw RNA-seq data ─────────────────────────────────────────────────
    rna_data = _load_rna_raw()
    rna_cols = [c for c in rna_data.columns if c != "SID"]
    rna_x_columns = {col: DataType.CONTINUOUS for col in rna_cols}
    rna_clin_x_columns = {**CLINICAL_X_COLUMNS, **rna_x_columns}

    all_summary, all_folds = [], []

    # ── Basic experiments (v1): raw counts, with/without grid search ──────────
    print("\n=== Basic experiments (raw RNA counts) ===")
    for use_gs, tag in [(False, "no_gs"), (True, "gs")]:
        gs_kwargs = {"param_grids": PARAM_GRIDS} if use_gs else {}

        # Exp 1: RNA → death
        rna_rad = rna_data.merge(radiomic_death, on="SID", how="inner")
        X_exp1 = rna_rad[rna_cols].reset_index(drop=True)
        y_exp1 = rna_rad["death"].reset_index(drop=True)

        if not use_gs:
            X_pre1 = build_preprocessor(rna_x_columns).fit_transform(X_exp1[~y_exp1.isna()])
            plot_pca_variance(X_pre1, title="PCA – RNA features (target: death)")
            plt.savefig(output_dir / "pca_variance_death.png", bbox_inches="tight", dpi=150)
            plt.close('all')

        cv1, fold1 = run_cv_experiment(
            X_exp1, y_exp1, rna_x_columns,
            label="RNA → death", n_splits=args.cv_folds, **gs_kwargs,
        )
        plot_cv_results(fold1, title=f"CV AUC – RNA → death ({tag})",
                        save_path=output_dir / f"cv_death_{tag}.png")
        all_summary.append(cv1)
        all_folds.append(fold1)

        # Exp 2: RNA → OS_central_event
        rna_clin = rna_data.merge(clinical_data[["SID", "OS_central_event"]], on="SID", how="inner")
        X_exp2 = rna_clin[rna_cols].reset_index(drop=True)
        y_exp2 = rna_clin["OS_central_event"].reset_index(drop=True)

        if not use_gs:
            X_pre2 = build_preprocessor(rna_x_columns).fit_transform(X_exp2[~y_exp2.isna()])
            plot_pca_variance(X_pre2, title="PCA – RNA features (target: OS_central_event)")
            plt.savefig(output_dir / "pca_variance_os.png", bbox_inches="tight", dpi=150)
            plt.close('all')

        cv2, fold2 = run_cv_experiment(
            X_exp2, y_exp2, rna_x_columns,
            label="RNA → OS_central_event", n_splits=args.cv_folds, **gs_kwargs,
        )
        plot_cv_results(fold2, title=f"CV AUC – RNA → OS_central_event ({tag})",
                        save_path=output_dir / f"cv_os_{tag}.png")
        all_summary.append(cv2)
        all_folds.append(fold2)

        # Exp 3: RNA + Clinical → death
        rna_clin_rad = rna_data.merge(radiomic_death, on="SID", how="inner").merge(
            clinical_data, on="SID", how="inner"
        )
        X_exp3 = rna_clin_rad[list(rna_clin_x_columns.keys())].reset_index(drop=True)
        y_exp3 = rna_clin_rad["death"].reset_index(drop=True)

        if not use_gs:
            X_pre3 = build_preprocessor(rna_clin_x_columns).fit_transform(X_exp3[~y_exp3.isna()])
            plot_pca_variance(X_pre3, title="PCA – RNA + clinical (target: death)")
            plt.savefig(output_dir / "pca_variance_rna_clin_death.png", bbox_inches="tight", dpi=150)
            plt.close('all')

        cv3, fold3 = run_cv_experiment(
            X_exp3, y_exp3, rna_clin_x_columns,
            label="RNA+Clinical → death", n_splits=args.cv_folds, **gs_kwargs,
        )
        plot_cv_results(fold3, title=f"CV AUC – RNA+Clinical → death ({tag})",
                        save_path=output_dir / f"cv_rna_clin_death_{tag}.png")
        all_summary.append(cv3)
        all_folds.append(fold3)

        # Exp 4: RNA + Clinical → OS_central_event
        rna_clin_full = rna_data.merge(clinical_data, on="SID", how="inner")
        X_exp4 = rna_clin_full[list(rna_clin_x_columns.keys())].reset_index(drop=True)
        y_exp4 = rna_clin_full["OS_central_event"].reset_index(drop=True)

        if not use_gs:
            X_pre4 = build_preprocessor(rna_clin_x_columns).fit_transform(X_exp4[~y_exp4.isna()])
            plot_pca_variance(X_pre4, title="PCA – RNA + clinical (target: OS_central_event)")
            plt.savefig(output_dir / "pca_variance_rna_clin_os.png", bbox_inches="tight", dpi=150)
            plt.close('all')

        cv4, fold4 = run_cv_experiment(
            X_exp4, y_exp4, rna_clin_x_columns,
            label="RNA+Clinical → OS_central_event", n_splits=args.cv_folds, **gs_kwargs,
        )
        plot_cv_results(fold4, title=f"CV AUC – RNA+Clinical → OS_central_event ({tag})",
                        save_path=output_dir / f"cv_rna_clin_os_{tag}.png")
        all_summary.append(cv4)
        all_folds.append(fold4)

        basic_folds = pd.concat([fold1, fold2, fold3, fold4], ignore_index=True)
        plot_experiment_comparison(basic_folds, title=f"RNA baselines — Test AUC ({tag})")
        plt.savefig(output_dir / f"experiment_comparison_{tag}.png", bbox_inches="tight", dpi=150)
        plt.close('all')

    # ── Pipeline comparison (v2): filtered log2(CPM) RNA ─────────────────────
    print("\n=== Pipeline comparison (filtered log2(CPM) RNA) ===")
    SELECT_K = args.select_k
    MAX_ITER = args.max_iter
    LR_C_SMALL = args.lr_c_small

    # Filter low-expression genes and normalise
    rna_raw_for_norm = _load_rna_raw()
    count_matrix = rna_raw_for_norm.set_index("SID")
    count_matrix = count_matrix.apply(pd.to_numeric, errors="coerce")
    min_samples = int(args.min_frac * count_matrix.shape[0])
    gene_mask = (count_matrix > args.min_count).sum(axis=0) >= min_samples
    count_filtered = count_matrix.loc[:, gene_mask]
    print(f"After filtering: {count_filtered.shape[1]} genes kept (removed {count_matrix.shape[1] - count_filtered.shape[1]})")

    library_sizes = count_filtered.sum(axis=1)
    cpm = count_filtered.div(library_sizes, axis=0) * 1e6
    rna_norm = np.log2(cpm + 1)
    rna_norm_data = rna_norm.rename_axis("SID").reset_index()
    norm_cols = [c for c in rna_norm_data.columns if c != "SID"]
    norm_x_columns = {col: DataType.CONTINUOUS for col in norm_cols}

    rna_death_norm = rna_norm_data.merge(radiomic_death, on="SID", how="inner")
    rna_os_norm = rna_norm_data.merge(clinical_data[["SID", "OS_central_event"]], on="SID", how="inner")
    X_death_n = rna_death_norm[norm_cols].reset_index(drop=True)
    y_death_n = rna_death_norm["death"].reset_index(drop=True)
    X_os_n = rna_os_norm[norm_cols].reset_index(drop=True)
    y_os_n = rna_os_norm["OS_central_event"].reset_index(drop=True)

    models_a = {
        "LR": LogisticRegression(solver="saga", l1_ratio=1.0, C=1.0, max_iter=MAX_ITER, random_state=RANDOM_STATE),
        "RF": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    }
    models_b = {
        f"LR (C={LR_C_SMALL})": LogisticRegression(
            solver="saga", l1_ratio=1.0, C=LR_C_SMALL, max_iter=MAX_ITER, random_state=RANDOM_STATE
        ),
    }

    pipe_folds = []

    # Exp A: SelectKBest + PCA
    cv_a1, fold_a1 = _run_pipeline_experiment(
        X_death_n, y_death_n, norm_x_columns,
        label=f"RNA(log2CPM)+KBest({SELECT_K}) → death",
        models=models_a, select_k=SELECT_K, n_splits=args.cv_folds,
    )
    plot_cv_results(fold_a1, title=f"Exp A: RNA(log2CPM)+KBest({SELECT_K}) → death",
                    save_path=output_dir / "pipeline_expA_death.png")
    pipe_folds.append(fold_a1)

    cv_a2, fold_a2 = _run_pipeline_experiment(
        X_os_n, y_os_n, norm_x_columns,
        label=f"RNA(log2CPM)+KBest({SELECT_K}) → OS_central_event",
        models=models_a, select_k=SELECT_K, n_splits=args.cv_folds,
    )
    plot_cv_results(fold_a2, title=f"Exp A: RNA(log2CPM)+KBest({SELECT_K}) → OS_central_event",
                    save_path=output_dir / "pipeline_expA_os.png")
    pipe_folds.append(fold_a2)

    # Exp B: SelectKBest + LR small C, no PCA
    cv_b1, fold_b1 = _run_pipeline_experiment(
        X_death_n, y_death_n, norm_x_columns,
        label=f"RNA(log2CPM)+KBest({SELECT_K})+L1(C={LR_C_SMALL}) → death",
        models=models_b, select_k=SELECT_K, use_pca=False, n_splits=args.cv_folds,
    )
    plot_cv_results(fold_b1, title=f"Exp B: KBest+L1(C={LR_C_SMALL}), no PCA → death",
                    save_path=output_dir / "pipeline_expB_death.png")
    pipe_folds.append(fold_b1)

    cv_b2, fold_b2 = _run_pipeline_experiment(
        X_os_n, y_os_n, norm_x_columns,
        label=f"RNA(log2CPM)+KBest({SELECT_K})+L1(C={LR_C_SMALL}) → OS_central_event",
        models=models_b, select_k=SELECT_K, use_pca=False, n_splits=args.cv_folds,
    )
    plot_cv_results(fold_b2, title=f"Exp B: KBest+L1(C={LR_C_SMALL}), no PCA → OS_central_event",
                    save_path=output_dir / "pipeline_expB_os.png")
    pipe_folds.append(fold_b2)

    all_pipe_folds = pd.concat(pipe_folds, ignore_index=True)
    plot_experiment_comparison(all_pipe_folds, title="RNA pipeline comparison — Test AUC")
    plt.savefig(output_dir / "pipeline_comparison.png", bbox_inches="tight", dpi=150)
    plt.close('all')

    pipeline_summary = pd.concat([cv_a1, cv_a2, cv_b1, cv_b2], ignore_index=True)
    print(f"\nConfig: SELECT_K={SELECT_K}, MAX_ITER={MAX_ITER}, LR_C_SMALL={LR_C_SMALL}")
    print(pipeline_summary.to_string(index=False))

    pd.concat(all_summary, ignore_index=True).to_csv(output_dir / "summary_basic.csv", index=False)
    pipeline_summary.to_csv(output_dir / "summary_pipeline.csv", index=False)
    pd.concat(all_folds, ignore_index=True).to_csv(output_dir / "fold_records_basic.csv", index=False)
    all_pipe_folds.to_csv(output_dir / "fold_records_pipeline.csv", index=False)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RNA-seq baseline for HCC outcome prediction (merged v1 + v2)."
    )
    parser.add_argument("--cv_folds", type=int, default=3,
                        help="Number of cross-validation folds (default: 3)")
    parser.add_argument("--select_k", type=int, default=500,
                        help="SelectKBest k for pipeline comparison (default: 500)")
    parser.add_argument("--lr_c_small", type=float, default=0.1,
                        help="LR C for strong L1 regularisation experiments (default: 0.1)")
    parser.add_argument("--max_iter", type=int, default=5000,
                        help="Max iterations for LogisticRegression (default: 5000)")
    parser.add_argument("--min_count", type=int, default=10,
                        help="Minimum count threshold for gene filtering (default: 10)")
    parser.add_argument("--min_frac", type=float, default=0.2,
                        help="Minimum fraction of samples above min_count for gene to pass (default: 0.2)")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (default: results/baseline/rna_baseline)")
    main(parser.parse_args())
