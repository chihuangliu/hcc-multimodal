"""Radiomic baseline: experiments on 446 pre-computed radiomic features (merged v1 + v2).

Experiment sets
---------------
basic   — Radiomic→death, Radiomic→OS, Clinical+Radiomic→OS; with/without grid search
pipeline — pipeline variants: SelectKBest+PCA, L1 regularisation (no grid search)
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


def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir) if args.output_dir else ROOT / "results" / "baseline" / "radiomic_baseline"
    output_dir.mkdir(parents=True, exist_ok=True)

    clinical_data = pd.read_csv(
        ROOT / "data" / "Clinical" / "2025_Nov_18_ICL_Resection_Clinical_Outcome_soramic_format.csv"
    ).dropna(how="all")

    radiomics_data = pd.read_csv(
        ROOT / "data" / "Radiomics" / "radiomic_cluster.csv"
    ).dropna(how="all")

    radiomic_death = radiomics_data[["SID", "death"]].copy()
    radiomic_features = radiomics_data.iloc[:, :447]
    radiomic_cols = [c for c in radiomic_features.columns if c != "SID"]
    radiomic_x_columns = {col: DataType.CONTINUOUS for col in radiomic_cols}
    combined_x_columns = {**CLINICAL_X_COLUMNS, **radiomic_x_columns}

    merged = clinical_data.merge(radiomic_features, on="SID", how="inner", suffixes=("_clin", "_rad"))
    print(f"Radiomic features: {len(radiomic_cols)}, patients: {len(radiomic_features)}")

    all_summary, all_folds = [], []

    # ── Basic experiments (v1): with and without grid search ─────────────────
    print("\n=== Basic experiments ===")
    for use_gs, tag in [(False, "no_gs"), (True, "gs")]:
        gs_kwargs = {"param_grids": PARAM_GRIDS} if use_gs else {}

        # Exp 1: Radiomic → death
        X_exp1 = radiomic_features.set_index("SID")[radiomic_cols]
        y_exp1 = radiomic_death.set_index("SID")["death"]
        common = X_exp1.index.intersection(y_exp1.index)
        X_exp1 = X_exp1.loc[common].reset_index(drop=True)
        y_exp1 = y_exp1.loc[common].reset_index(drop=True)

        if not use_gs:
            X_pre1 = build_preprocessor(radiomic_x_columns).fit_transform(X_exp1[~y_exp1.isna()])
            plot_pca_variance(X_pre1, title="PCA – radiomic features (target: death)")
            plt.savefig(output_dir / "pca_variance_death.png", bbox_inches="tight", dpi=150)
            plt.close('all')

        cv1, fold1 = run_cv_experiment(
            X_exp1, y_exp1, radiomic_x_columns,
            label="Radiomic → death",
            n_splits=args.cv_folds, **gs_kwargs,
        )
        plot_cv_results(fold1, title=f"CV AUC – Radiomic → death ({tag})",
                        save_path=output_dir / f"cv_death_{tag}.png")
        all_summary.append(cv1)
        all_folds.append(fold1)

        # Exp 2: Radiomic → OS_central_event
        rad_with_clin_y = radiomic_features.merge(
            clinical_data[["SID", "OS_central_event"]], on="SID", how="inner"
        )
        X_exp2 = rad_with_clin_y[radiomic_cols].reset_index(drop=True)
        y_exp2 = rad_with_clin_y["OS_central_event"].reset_index(drop=True)

        if not use_gs:
            X_pre2 = build_preprocessor(radiomic_x_columns).fit_transform(X_exp2[~y_exp2.isna()])
            plot_pca_variance(X_pre2, title="PCA – radiomic features (target: OS_central_event)")
            plt.savefig(output_dir / "pca_variance_os.png", bbox_inches="tight", dpi=150)
            plt.close('all')

        cv2, fold2 = run_cv_experiment(
            X_exp2, y_exp2, radiomic_x_columns,
            label="Radiomic → OS_central_event",
            n_splits=args.cv_folds, **gs_kwargs,
        )
        plot_cv_results(fold2, title=f"CV AUC – Radiomic → OS_central_event ({tag})",
                        save_path=output_dir / f"cv_os_{tag}.png")
        all_summary.append(cv2)
        all_folds.append(fold2)

        # Exp 3: Clinical + Radiomic → OS_central_event
        X_exp3 = merged[list(combined_x_columns.keys())]
        y_exp3 = merged["OS_central_event"]

        if not use_gs:
            X_pre3 = build_preprocessor(combined_x_columns).fit_transform(X_exp3[~y_exp3.isna()])
            plot_pca_variance(X_pre3, title="PCA – clinical + radiomic (target: OS_central_event)")
            plt.savefig(output_dir / "pca_variance_clin_rad_os.png", bbox_inches="tight", dpi=150)
            plt.close('all')

        cv3, fold3 = run_cv_experiment(
            X_exp3, y_exp3, combined_x_columns,
            label="Clinical+Radiomic → OS_central_event",
            n_splits=args.cv_folds, **gs_kwargs,
        )
        plot_cv_results(fold3, title=f"CV AUC – Clinical+Radiomic → OS_central_event ({tag})",
                        save_path=output_dir / f"cv_clin_rad_os_{tag}.png")
        all_summary.append(cv3)
        all_folds.append(fold3)

        all_basic_folds = pd.concat([fold1, fold2, fold3], ignore_index=True)
        plot_experiment_comparison(all_basic_folds,
                                   title=f"Radiomic baselines — Test AUC ({tag})")
        plt.savefig(output_dir / f"experiment_comparison_{tag}.png", bbox_inches="tight", dpi=150)
        plt.close('all')

    # ── Pipeline comparison experiments (v2) ─────────────────────────────────
    print("\n=== Pipeline comparison experiments ===")
    SELECT_K = args.select_k
    MAX_ITER = args.max_iter
    LR_C_SMALL = args.lr_c_small

    models_default = {
        "LR": LogisticRegression(solver="saga", l1_ratio=1.0, C=1.0, max_iter=MAX_ITER, random_state=RANDOM_STATE),
        "RF": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    }
    models_small_c = {
        f"LR (C={LR_C_SMALL})": LogisticRegression(
            solver="saga", l1_ratio=1.0, C=LR_C_SMALL, max_iter=MAX_ITER, random_state=RANDOM_STATE
        ),
    }

    X_death = radiomic_features.set_index("SID")[radiomic_cols]
    y_death = radiomic_death.set_index("SID")["death"]
    common = X_death.index.intersection(y_death.index)
    X_death = X_death.loc[common].reset_index(drop=True)
    y_death = y_death.loc[common].reset_index(drop=True)

    rad_os = radiomic_features.merge(clinical_data[["SID", "OS_central_event"]], on="SID", how="inner")
    X_os = rad_os[radiomic_cols].reset_index(drop=True)
    y_os = rad_os["OS_central_event"].reset_index(drop=True)

    pipe_folds = []

    # Exp 2: SelectKBest + PCA
    cv_2a, fold_2a = _run_pipeline_experiment(
        X_death, y_death, radiomic_x_columns,
        label=f"Radiomic+KBest({SELECT_K}) → death",
        models=models_default, select_k=SELECT_K, n_splits=args.cv_folds,
    )
    plot_cv_results(fold_2a, title=f"Exp 2: Radiomic+KBest({SELECT_K}) → death",
                    save_path=output_dir / "pipeline_exp2_death.png")
    pipe_folds.append(fold_2a)

    cv_2b, fold_2b = _run_pipeline_experiment(
        X_os, y_os, radiomic_x_columns,
        label=f"Radiomic+KBest({SELECT_K}) → OS_central_event",
        models=models_default, select_k=SELECT_K, n_splits=args.cv_folds,
    )
    plot_cv_results(fold_2b, title=f"Exp 2: Radiomic+KBest({SELECT_K}) → OS_central_event",
                    save_path=output_dir / "pipeline_exp2_os.png")
    pipe_folds.append(fold_2b)

    # Exp 3: No SelectKBest, no PCA, LR small C
    cv_3a, fold_3a = _run_pipeline_experiment(
        X_death, y_death, radiomic_x_columns,
        label=f"Radiomic+L1(C={LR_C_SMALL}) → death",
        models=models_small_c, use_pca=False, n_splits=args.cv_folds,
    )
    plot_cv_results(fold_3a, title=f"Exp 3: Radiomic+L1(C={LR_C_SMALL}), no PCA → death",
                    save_path=output_dir / "pipeline_exp3_death.png")
    pipe_folds.append(fold_3a)

    cv_3b, fold_3b = _run_pipeline_experiment(
        X_os, y_os, radiomic_x_columns,
        label=f"Radiomic+L1(C={LR_C_SMALL}) → OS_central_event",
        models=models_small_c, use_pca=False, n_splits=args.cv_folds,
    )
    plot_cv_results(fold_3b, title=f"Exp 3: Radiomic+L1(C={LR_C_SMALL}), no PCA → OS_central_event",
                    save_path=output_dir / "pipeline_exp3_os.png")
    pipe_folds.append(fold_3b)

    # Exp 4: SelectKBest, no PCA, LR small C
    cv_4a, fold_4a = _run_pipeline_experiment(
        X_death, y_death, radiomic_x_columns,
        label=f"Radiomic+KBest({SELECT_K})+L1(C={LR_C_SMALL}) → death",
        models=models_small_c, select_k=SELECT_K, use_pca=False, n_splits=args.cv_folds,
    )
    plot_cv_results(fold_4a, title=f"Exp 4: KBest({SELECT_K})+L1(C={LR_C_SMALL}), no PCA → death",
                    save_path=output_dir / "pipeline_exp4_death.png")
    pipe_folds.append(fold_4a)

    cv_4b, fold_4b = _run_pipeline_experiment(
        X_os, y_os, radiomic_x_columns,
        label=f"Radiomic+KBest({SELECT_K})+L1(C={LR_C_SMALL}) → OS_central_event",
        models=models_small_c, select_k=SELECT_K, use_pca=False, n_splits=args.cv_folds,
    )
    plot_cv_results(fold_4b, title=f"Exp 4: KBest({SELECT_K})+L1(C={LR_C_SMALL}), no PCA → OS_central_event",
                    save_path=output_dir / "pipeline_exp4_os.png")
    pipe_folds.append(fold_4b)

    all_pipe_folds = pd.concat(pipe_folds, ignore_index=True)
    plot_experiment_comparison(all_pipe_folds, title="Pipeline comparison — Test AUC")
    plt.savefig(output_dir / "pipeline_comparison.png", bbox_inches="tight", dpi=150)
    plt.close('all')

    # Save summaries
    pipeline_summary = pd.concat([cv_2a, cv_2b, cv_3a, cv_3b, cv_4a, cv_4b], ignore_index=True)
    print(f"\nConfig: SELECT_K={SELECT_K}, MAX_ITER={MAX_ITER}, LR_C_SMALL={LR_C_SMALL}")
    print(pipeline_summary.to_string(index=False))

    pd.concat(all_summary, ignore_index=True).to_csv(output_dir / "summary_basic.csv", index=False)
    pipeline_summary.to_csv(output_dir / "summary_pipeline.csv", index=False)
    pd.concat(all_folds, ignore_index=True).to_csv(output_dir / "fold_records_basic.csv", index=False)
    all_pipe_folds.to_csv(output_dir / "fold_records_pipeline.csv", index=False)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Radiomic baseline for HCC outcome prediction (merged v1 + v2)."
    )
    parser.add_argument("--cv_folds", type=int, default=3,
                        help="Number of cross-validation folds (default: 3)")
    parser.add_argument("--select_k", type=int, default=100,
                        help="SelectKBest k for pipeline comparison (default: 100)")
    parser.add_argument("--lr_c_small", type=float, default=0.1,
                        help="LR C for strong L1 regularisation experiments (default: 0.1)")
    parser.add_argument("--max_iter", type=int, default=5000,
                        help="Max iterations for LogisticRegression (default: 5000)")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (default: results/baseline/radiomic_baseline)")
    main(parser.parse_args())
