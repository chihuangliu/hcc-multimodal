"""Cross-validation evaluation utilities for HCC multimodal baselines."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from hcc_multimodal.baselines.config import RANDOM_STATE, MODELS
from hcc_multimodal.baselines.transforms import build_preprocessor


def run_cv_experiment(
    X: pd.DataFrame,
    y: pd.Series,
    x_columns: dict,
    label: str,
    pca_n_components: float = 0.9,
    n_splits: int = 3,
    param_grids: dict[str, dict] | None = None,
    models: dict[str, object] = MODELS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run stratified k-fold CV for LR and RF with PCA preprocessing.

    Parameters
    ----------
    X:
        Feature DataFrame. Column names must match ``x_columns`` keys.
    y:
        Target Series. Rows where ``y`` is NaN are dropped before fitting.
    x_columns:
        Mapping of column name → :class:`~hcc_multimodal.baselines.transforms.DataType`.
    label:
        Experiment label written into the returned DataFrames.
    n_splits:
        Number of CV folds.
    param_grids:
        Per-model hyperparameter grids for inner-CV grid search. Pass ``None``
        to skip grid search and evaluate the pipeline with default parameters.
    models:
        Mapping of model name → estimator instance. Defaults to LR + RF.

    Returns
    -------
    summary_df:
        One row per model with mean/std AUC and accuracy.
    fold_df:
        One row per (model, fold) with train and test AUC.
    """
    preprocessor = build_preprocessor(x_columns)
    outer_cv = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE
    )
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    mask = ~y.isna()
    X_clean, y_clean = X[mask], y[mask]
    print(
        f"[{label}]  n={len(y_clean)}"
        f"  positives={int(y_clean.sum())}"
        f"  features={X_clean.shape[1]}"
    )

    records, fold_records = [], []
    for model_name, model in models.items():
        pipe = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("pca", PCA(n_components=pca_n_components, random_state=RANDOM_STATE)),
                ("model", model),
            ]
        )
        if param_grids is not None:
            estimator = GridSearchCV(
                pipe,
                param_grids[model_name],
                cv=inner_cv,
                scoring="roc_auc",
                refit=True,
                n_jobs=-1,
            )
        else:
            estimator = pipe

        scores = cross_validate(
            estimator,
            X_clean,
            y_clean,
            cv=outer_cv,
            scoring=["roc_auc", "accuracy"],
            return_train_score=True,
            return_estimator=True,
        )

        records.append(
            {
                "experiment": label,
                "model": model_name,
                "AUC mean": round(scores["test_roc_auc"].mean(), 3),
                "AUC std": round(scores["test_roc_auc"].std(), 3),
                "Accuracy mean": round(scores["test_accuracy"].mean(), 3),
                "Accuracy std": round(scores["test_accuracy"].std(), 3),
            }
        )
        for fold_i, (est, train_auc, test_auc) in enumerate(
            zip(scores["estimator"], scores["train_roc_auc"], scores["test_roc_auc"])
        ):
            best_params = (
                {k.replace("model__", ""): v for k, v in est.best_params_.items()}
                if param_grids is not None
                else {}
            )
            fold_records.append(
                {
                    "experiment": label,
                    "model": model_name,
                    "fold": fold_i + 1,
                    "train_auc": train_auc,
                    "test_auc": test_auc,
                    "best_params": best_params,
                }
            )
    return pd.DataFrame(records), pd.DataFrame(fold_records)


def plot_pca_variance(
    X_pre: np.ndarray,
    title: str = "PCA on preprocessed features",
    threshold: float = 0.90,
) -> None:
    """Plot cumulative and per-component explained variance from PCA."""
    n_components = min(X_pre.shape)
    pca_full = PCA(n_components=n_components, random_state=RANDOM_STATE).fit(X_pre)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    percentile = int(np.searchsorted(cumvar, threshold)) + 1

    fig, axes = plt.subplots(1, 2, figsize=(10, 3))

    ax = axes[0]
    ax.plot(np.arange(1, len(cumvar) + 1), cumvar, color="steelblue", linewidth=1.5)
    ax.axhline(threshold, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(
        percentile,
        color="tomato",
        linestyle="--",
        linewidth=0.8,
        label=f"{percentile} components → {threshold:.0%} variance",
    )
    ax.set_xlabel("Number of PCA components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title("Cumulative variance explained")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.bar(np.arange(1, 31), pca_full.explained_variance_ratio_[:30], color="steelblue")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title("Per-component variance (first 30)")
    ax.grid(axis="y", alpha=0.3)

    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.show()
    print(
        f"Components to reach {threshold:.0%} variance: {percentile} / {X_pre.shape[1]}"
    )


def plot_cv_results(
    fold_df: pd.DataFrame,
    title: str = "Cross-validation AUC",
) -> None:
    """Strip + diamond plot of per-fold train/test AUC."""
    model_names = fold_df["model"].unique().tolist()
    colors = ["steelblue", "seagreen"]
    y_pos = np.arange(len(model_names))

    fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
    for col, split in enumerate(["train", "test"]):
        ax = axes[col]
        for y_idx, (model_name, color) in enumerate(zip(model_names, colors)):
            df = fold_df[fold_df["model"] == model_name]
            aucs = df[f"{split}_auc"].values
            ax.scatter(aucs, [y_idx] * len(aucs), color=color, s=60, zorder=3)
            ax.scatter(aucs.mean(), y_idx, marker="D", color=color, s=90, zorder=4)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(model_names)
        ax.set_xlabel("AUC")
        ax.set_title(f"{'Train' if split == 'train' else 'Test'} AUC")
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_experiment_comparison(
    all_folds: pd.DataFrame,
    title: str = "Test AUC comparison across experiments",
) -> None:
    """Side-by-side strip plot of test AUC for every experiment."""
    experiments = all_folds["experiment"].unique().tolist()
    model_names = all_folds["model"].unique().tolist()
    colors = ["steelblue", "seagreen"]

    fig, axes = plt.subplots(
        1, len(experiments), figsize=(5 * len(experiments), 3), sharey=False
    )
    if len(experiments) == 1:
        axes = [axes]

    for ax, exp_name in zip(axes, experiments):
        df_exp = all_folds[all_folds["experiment"] == exp_name]
        y_pos = np.arange(len(model_names))
        for y_idx, (model_name, color) in enumerate(zip(model_names, colors)):
            aucs = df_exp[df_exp["model"] == model_name]["test_auc"].values
            ax.scatter(aucs, [y_idx] * len(aucs), color=color, s=60, zorder=3)
            ax.scatter(
                aucs.mean(),
                y_idx,
                marker="D",
                color=color,
                s=90,
                zorder=4,
                label=model_name,
            )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(model_names)
        ax.set_xlabel("Test AUC")
        ax.set_title(exp_name, fontsize=9)
        ax.grid(axis="x", alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=9)
    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.show()
