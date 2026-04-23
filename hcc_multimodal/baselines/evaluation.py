"""Cross-validation evaluation utilities for HCC multimodal baselines."""

import warnings
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectorMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from hcc_multimodal.baselines.config import RANDOM_STATE, MODELS
from hcc_multimodal.baselines.transforms import DataType, build_preprocessor


class DeseqFeatureSelector(SelectorMixin, BaseEstimator):
    """DESeq2-based feature selector for raw RNA-seq count data.

    Fits a DESeq2 model inside each CV fold (no data leakage) and selects
    genes whose Benjamini-Hochberg-adjusted p-value is below ``pvalue``.

    Parameters
    ----------
    pvalue:
        Adjusted p-value threshold for gene selection.
    """

    def __init__(self, pvalue: float = 0.05):
        self.pvalue = pvalue

    def fit(self, X: pd.DataFrame, y) -> "DeseqFeatureSelector":
        metadata = pd.DataFrame({"condition": np.asarray(y, dtype=int)}, index=X.index)
        dds = DeseqDataSet(counts=X, metadata=metadata, design="~condition")
        dds.deseq2()
        stat_res = DeseqStats(dds, contrast=["condition", 1, 0], alpha=self.pvalue)
        stat_res.summary()
        self.pvalues_ = stat_res.results_df["padj"].fillna(1.0).values
        self.scores_ = -np.log10(np.clip(self.pvalues_, 1e-300, 1.0))
        self.n_features_in_ = X.shape[1]
        return self

    def _get_support_mask(self) -> np.ndarray:
        check_is_fitted(self, "pvalues_")
        return self.pvalues_ < self.pvalue

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        mask = self._get_support_mask()
        return X.loc[:, mask] if isinstance(X, pd.DataFrame) else X[:, mask]


class DeseqCPMSelector(BaseEstimator, TransformerMixin):
    """DESeq2 gene selection + log2(CPM) normalization for raw RNA-seq counts.

    Must be the first step of the pipeline — both DESeq2 and the
    library-size computation require raw integer counts. At ``fit`` time,
    a DESeq2 model is fit on the training fold and the Benjamini-Hochberg
    adjusted p-value mask is stored. At ``transform`` time, each sample's
    library size is computed from the full (pre-selection) input, the
    selected genes are subset, and values are returned as
    ``log2(counts / library_size * 1e6 + pseudocount)``.

    Parameters
    ----------
    pvalue:
        Adjusted p-value threshold for gene selection.
    pseudocount:
        Added inside the log to keep zero-count entries finite.
    min_features:
        If fewer than this many genes pass the p-value threshold in a
        fold, fall back to the top ``min_features`` genes ranked by
        padj. Guards against folds where no gene survives BH correction
        (common with small n and high-dim RNA-seq).
    """

    def __init__(
        self,
        pvalue: float = 0.05,
        pseudocount: float = 1.0,
        min_features: int = 20,
    ):
        self.pvalue = pvalue
        self.pseudocount = pseudocount
        self.min_features = min_features

    def fit(self, X: pd.DataFrame, y) -> "DeseqCPMSelector":
        metadata = pd.DataFrame({"condition": np.asarray(y, dtype=int)}, index=X.index)
        dds = DeseqDataSet(counts=X, metadata=metadata, design="~condition")
        dds.deseq2()
        stat_res = DeseqStats(dds, contrast=["condition", 1, 0], alpha=self.pvalue)
        stat_res.summary()
        padj = stat_res.results_df["padj"].fillna(1.0).values
        mask = padj < self.pvalue
        if mask.sum() < self.min_features:
            k = min(self.min_features, len(padj))
            warnings.warn(
                f"DeseqCPMSelector: only {int(mask.sum())} gene(s) passed "
                f"padj<{self.pvalue}; falling back to top {k} by padj.",
                stacklevel=2,
            )
            top_idx = np.argsort(padj, kind="stable")[:k]
            mask = np.zeros_like(padj, dtype=bool)
            mask[top_idx] = True
        self.support_ = mask
        self.padj_ = padj
        self.n_features_in_ = X.shape[1]
        if hasattr(X, "columns"):
            self.feature_names_in_ = X.columns.to_numpy()
        return self

    def get_support(self, indices: bool = False) -> np.ndarray:
        check_is_fitted(self, "support_")
        return np.where(self.support_)[0] if indices else self.support_

    def transform(self, X):
        check_is_fitted(self, "support_")
        library_size = np.asarray(X.sum(axis=1), dtype=float).reshape(-1)
        library_size = np.where(library_size <= 0, 1.0, library_size)
        if isinstance(X, pd.DataFrame):
            X_sel = X.loc[:, self.support_]
            cpm = X_sel.div(library_size, axis=0) * 1e6
        else:
            X_sel = X[:, self.support_]
            cpm = X_sel / library_size[:, None] * 1e6
        return np.log2(cpm + self.pseudocount)


def run_cv_experiment(
    X: pd.DataFrame,
    y: pd.Series,
    x_columns: dict,
    label: str,
    pca_n_components: float | None = None,
    n_splits: int = 3,
    param_grids: dict[str, dict] | None = None,
    models: dict[str, object] = MODELS,
    feature_selector: SelectorMixin | None = None,
    selector_first: bool = False,
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
    selector_first:
        If True, place ``feature_selector`` as the first pipeline step and
        skip the ``x_columns``-based preprocessor. Use this when the
        selector needs raw inputs (e.g. :class:`DeseqCPMSelector` on raw
        RNA-seq counts) — the selector is then responsible for any
        normalization.

    Returns
    -------
    summary_df:
        One row per model with mean/std AUC and accuracy.
    fold_df:
        One row per (model, fold) with train and test AUC.
    """
    if feature_selector is not None and selector_first:
        non_continuous = {
            col: t for col, t in x_columns.items() if t != DataType.CONTINUOUS
        }
        if non_continuous:
            raise ValueError(
                "selector_first=True only supports continuous features "
                "(the post-selector preprocessor is imputer + scaler). "
                f"Non-continuous columns in x_columns: {non_continuous}"
            )
        preprocessor = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
    else:
        preprocessor = build_preprocessor(x_columns)
    outer_cv = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE
    )
    inner_cv = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE
    )

    mask = ~y.isna()
    X_clean, y_clean = X[mask], y[mask]
    print(
        f"[{label}]  n={len(y_clean)}"
        f"  positives={int(y_clean.sum())}"
        f"  features={X_clean.shape[1]}"
    )

    records, fold_records = [], []
    for model_name, model in models.items():
        steps = []
        if feature_selector is not None and selector_first:
            steps.append(("feature_selection", feature_selector))
            steps.append(("preprocessor", preprocessor))
        else:
            steps.append(("preprocessor", preprocessor))
            if feature_selector is not None:
                steps.append(("feature_selection", feature_selector))
        if pca_n_components is not None:
            steps.append(
                ("pca", PCA(n_components=pca_n_components, random_state=RANDOM_STATE))
            )
        steps.append(("model", model))
        pipe = Pipeline(steps)
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
            error_score="raise",
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
            sel = est.named_steps.get("feature_selection") if hasattr(est, "named_steps") else None
            if sel is not None and hasattr(sel, "feature_names_in_"):
                selected_features = pd.DataFrame({
                    "gene": sel.feature_names_in_[sel.support_],
                    "padj": sel.padj_[sel.support_],
                }).sort_values("padj").reset_index(drop=True)
            else:
                selected_features = None
            fold_records.append(
                {
                    "experiment": label,
                    "model": model_name,
                    "fold": fold_i + 1,
                    "train_auc": train_auc,
                    "test_auc": test_auc,
                    "best_params": best_params,
                    "selected_features": selected_features,
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
    save_path=None,
) -> None:
    """Strip + diamond plot of per-fold train/test AUC."""
    model_names = fold_df["model"].unique().tolist()
    colors = ["steelblue", "seagreen"]
    y_pos = np.arange(len(model_names))

    fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
    for col, split in enumerate(["train", "test"]):
        ax = axes[col]
        for y_idx, (model_name, color) in enumerate(zip(model_names, cycle(colors))):
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
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
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
