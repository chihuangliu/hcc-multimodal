"""Cross-validation evaluation utilities for HCC multimodal baselines."""

import warnings
from itertools import cycle
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams['svg.fonttype'] = 'none'
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectorMixin
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
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


def apply_selector_before_cv(
    X: pd.DataFrame,
    y: pd.Series,
    x_columns: dict,
    selector,
    selector_first: bool = False,
    rna_cpm: bool = False,
    label: str = "",
    save_path=None,
) -> tuple:
    """Fit a feature selector on the full labelled set before CV (leaks labels — diagnostic only).

    Returns pre-transformed X, filtered y, updated x_columns, and cv_kwargs
    ready to unpack into run_cv_experiment (feature_selector=None so no
    second selection happens inside the folds).

    Parameters
    ----------
    selector_first:
        When True the selector receives raw X (e.g. DeseqCPMSelector on counts).
        When False the column-type preprocessor runs first.
    save_path:
        If given, saves a CSV of selected feature names to this path.
    """
    y_mask = ~y.isna()
    y_fit = y[y_mask]
    X_fit = X.loc[y_fit.index]
    sel_pre = clone(selector)

    if selector_first:
        X_arr = sel_pre.fit_transform(X_fit, y_fit)
        support = (
            sel_pre.support_
            if hasattr(sel_pre, "support_")
            else sel_pre.get_support()
        )
        feature_names = np.asarray(sel_pre.feature_names_in_)[support]
    else:
        # Use CPMTransformer for RNA counts; simple imputer+scaler for everything else.
        # Both preserve column count/order, so input column names index the support mask.
        if rna_cpm:
            prep_pre = build_preprocessor(x_columns, rna_cpm=True)
        else:
            prep_pre = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )
        X_pre = prep_pre.fit_transform(X_fit)
        X_arr = sel_pre.fit_transform(X_pre, y_fit)
        support = sel_pre.get_support()
        all_names = (
            np.asarray(X_fit.columns)
            if hasattr(X_fit, "columns")
            else np.array([f"x{i}" for i in range(X_fit.shape[1])])
        )
        feature_names = all_names[support]

    X_cv = pd.DataFrame(np.asarray(X_arr), columns=feature_names, index=X_fit.index)
    y_cv = y_fit
    x_columns_cv = {c: DataType.CONTINUOUS for c in feature_names}
    cv_kwargs = dict(feature_selector=None, selector_first=False, rna_cpm=False)

    if label:
        print(f"[{label}] selector applied before CV: {len(feature_names)} features")
    if save_path is not None:
        pd.DataFrame({"feature": feature_names}).to_csv(save_path, index=False)

    return X_cv, y_cv, x_columns_cv, cv_kwargs


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
    rna_cpm: bool = False,
    return_proba: bool = False,
    X_confound: "pd.DataFrame | None" = None,
    confound_x_columns: "dict | None" = None,
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
    rna_cpm:
        Passed to :func:`~hcc_multimodal.baselines.transforms.build_preprocessor`.
        When True, the preprocessor is ``CPMTransformer → StandardScaler``
        instead of the column-type ColumnTransformer. Use this for the
        SelectKBest path so the model sees log2(CPM)-normalised features,
        matching what :class:`DeseqCPMSelector` produces.

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
        preprocessor = build_preprocessor(x_columns, rna_cpm=rna_cpm)
    outer_cv = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE
    )
    inner_cv = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE
    )

    mask = ~y.isna()
    y_clean = y[mask]
    X_clean = X.loc[y_clean.index]
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

        # Confounders are concatenated after the feature pipeline (preprocessor +
        # optional selector) inside each fold, so they never enter the selector
        # while still being available to the model. Works for both before-CV and
        # in-CV selection paths.
        use_confound_path = X_confound is not None
        if use_confound_path:
            feat_steps_no_model = [(n, e) for n, e in steps if n != "model"]
            feat_pipe_template = Pipeline(feat_steps_no_model)
            conf_pipe_template = build_preprocessor(confound_x_columns or {})
            X_confound_aligned = X_confound.loc[X_clean.index]

        fold_aucs_train, fold_aucs_test, fold_accs_test = [], [], []
        for fold_i, (train_idx, test_idx) in enumerate(outer_cv.split(X_clean, y_clean)):
            X_tr = X_clean.iloc[train_idx]
            X_te = X_clean.iloc[test_idx]
            y_tr = y_clean.iloc[train_idx]
            y_te = y_clean.iloc[test_idx]

            if use_confound_path:
                fp = clone(feat_pipe_template)
                X_tr_feat = fp.fit_transform(X_tr, y_tr)
                X_te_feat = fp.transform(X_te)

                conf_tr = X_confound_aligned.iloc[train_idx]
                conf_te = X_confound_aligned.iloc[test_idx]
                cp = clone(conf_pipe_template)
                X_tr_combined = np.hstack([X_tr_feat, cp.fit_transform(conf_tr)])
                X_te_combined = np.hstack([X_te_feat, cp.transform(conf_te)])

                est = clone(model)
                est.fit(X_tr_combined, y_tr)
                te_proba = est.predict_proba(X_te_combined)[:, 1]
                tr_proba = est.predict_proba(X_tr_combined)[:, 1]
                test_acc = accuracy_score(y_te, est.predict(X_te_combined))
            else:
                est = clone(estimator)
                est.fit(X_tr, y_tr)
                te_proba = est.predict_proba(X_te)[:, 1]
                tr_proba = est.predict_proba(X_tr)[:, 1]
                test_acc = accuracy_score(y_te, est.predict(X_te))

            train_auc = roc_auc_score(y_tr, tr_proba)
            test_auc = roc_auc_score(y_te, te_proba)

            fold_aucs_train.append(train_auc)
            fold_aucs_test.append(test_auc)
            fold_accs_test.append(test_acc)

            best_params = (
                {k.replace("model__", ""): v for k, v in est.best_params_.items()}
                if param_grids is not None and not use_confound_path
                else {}
            )
            sel = est.named_steps.get("feature_selection") if hasattr(est, "named_steps") else None
            preproc = est.named_steps.get("preprocessor") if hasattr(est, "named_steps") else None
            selected_features = None
            selected_names = None
            if sel is not None:
                support = sel.support_ if hasattr(sel, "support_") else sel.get_support()
                if hasattr(sel, "feature_names_in_"):
                    selected_names = np.asarray(sel.feature_names_in_)[support]
                elif preproc is not None and hasattr(preproc, "get_feature_names_out"):
                    try:
                        preproc_names = np.asarray(preproc.get_feature_names_out())
                        preproc_names = np.array(
                            [n.split("__", 1)[1] if "__" in n else n for n in preproc_names]
                        )
                        selected_names = preproc_names[support]
                    except Exception:
                        selected_names = None
                if selected_names is not None:
                    if hasattr(sel, "padj_"):
                        selected_features = pd.DataFrame({
                            "feature": selected_names,
                            "padj": sel.padj_[support],
                        }).sort_values("padj").reset_index(drop=True)
                    elif hasattr(sel, "pvalues_") and sel.pvalues_ is not None:
                        selected_features = pd.DataFrame({
                            "feature": selected_names,
                            "pvalue": sel.pvalues_[support],
                        }).sort_values("pvalue").reset_index(drop=True)
                    else:
                        selected_features = pd.DataFrame({"feature": selected_names})
            elif preproc is not None and hasattr(preproc, "get_feature_names_out"):
                try:
                    preproc_names = np.asarray(preproc.get_feature_names_out())
                    preproc_names = np.array(
                        [n.split("__", 1)[1] if "__" in n else n for n in preproc_names]
                    )
                    selected_features = pd.DataFrame({"feature": preproc_names})
                except Exception:
                    selected_features = None

            lr_nonzero_features = None
            fitted_model = est.named_steps.get("model") if hasattr(est, "named_steps") else None
            if (
                fitted_model is not None
                and hasattr(fitted_model, "coef_")
                and selected_features is not None
            ):
                coef = fitted_model.coef_[0]
                nonzero_mask = np.abs(coef) > 0
                lr_nonzero_features = pd.DataFrame({
                    "feature": selected_features["feature"].values[nonzero_mask],
                    "coefficient": coef[nonzero_mask],
                }).sort_values("coefficient", key=np.abs, ascending=False).reset_index(drop=True)

            fold_record = {
                "experiment": label,
                "model": model_name,
                "fold": fold_i + 1,
                "train_auc": train_auc,
                "test_auc": test_auc,
                "best_params": best_params,
                "selected_features": selected_features,
                "lr_nonzero_features": lr_nonzero_features,
            }
            if return_proba:
                fold_record["test_indices"] = X_clean.index[test_idx].tolist()
                fold_record["test_proba"] = te_proba.tolist()
            fold_records.append(fold_record)

        records.append(
            {
                "experiment": label,
                "model": model_name,
                "AUC mean": round(np.mean(fold_aucs_test), 3),
                "AUC std": round(np.std(fold_aucs_test), 3),
                "Accuracy mean": round(np.mean(fold_accs_test), 3),
                "Accuracy std": round(np.std(fold_accs_test), 3),
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

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

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
    ax.set_xlabel("Number of Principal Components", fontsize=16)
    ax.set_ylabel("Cumulative Explained Variance", fontsize=16)
    ax.set_title("Cumulative Variance Explained", fontsize=16)
    ax.tick_params(labelsize=14)
    ax.legend(fontsize=14)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.bar(np.arange(1, 31), pca_full.explained_variance_ratio_[:30], color="steelblue")
    ax.set_xlabel("Principal Component", fontsize=16)
    ax.set_ylabel("Explained Variance Ratio", fontsize=16)
    ax.set_title("Per-component Variance (first 30)", fontsize=16)
    ax.tick_params(labelsize=14)
    ax.grid(axis="y", alpha=0.3)

    plt.suptitle(title, fontsize=18)
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

    fig, axes = plt.subplots(1, 2, figsize=(10, max(3, len(model_names) * 0.6 + 1)), sharey=True)
    for col, split in enumerate(["train", "test"]):
        ax = axes[col]
        for y_idx, (model_name, color) in enumerate(zip(model_names, cycle(colors))):
            df = fold_df[fold_df["model"] == model_name]
            aucs = df[f"{split}_auc"].values
            ax.scatter(aucs, [y_idx] * len(aucs), color=color, s=80, zorder=3)
            ax.scatter(aucs.mean(), y_idx, marker="D", color=color, s=110, zorder=4)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(model_names, fontsize=14)
        ax.set_xlabel("Area Under the ROC Curve (AUC)", fontsize=16)
        ax.set_title(f"{'Train' if split == 'train' else 'Test'} AUC", fontsize=16)
        ax.tick_params(axis="x", labelsize=14)
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle(title, fontsize=18)
    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        fig.savefig(save_path.with_suffix(".svg"), bbox_inches="tight")
        plt.close(fig)
    else:
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
            ax.scatter(aucs, [y_idx] * len(aucs), color=color, s=80, zorder=3)
            ax.scatter(
                aucs.mean(),
                y_idx,
                marker="D",
                color=color,
                s=110,
                zorder=4,
                label=model_name,
            )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(model_names, fontsize=14)
        ax.set_xlabel("Test AUROC", fontsize=16)
        ax.set_title(exp_name, fontsize=14)
        ax.tick_params(axis="x", labelsize=14)
        ax.grid(axis="x", alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=14)
    fig.suptitle(title, fontsize=18)
    plt.tight_layout()
    plt.show()
