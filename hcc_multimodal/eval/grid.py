"""Feature-selection × classifier zoos for the embedding benchmark grid.

Reproduces the Radiomics "BvM — AUC" benchmark heatmap (10 classifiers × 12
feature-selection techniques), but applied to contrastive *embeddings* rather than
radiomic features. Shared by ``scripts/embedding_grid_eval.py`` and the survival
grid extension so the two stay in lockstep.

Design notes
------------
* ``MODEL_ZOO[name]`` = ``(estimator, param_grid)`` where ``param_grid`` keys are
  ``model__*`` (ready for ``GridSearchCV`` over a ``build_grid_pipeline`` pipeline).
* ``LR`` is unregularized logistic regression; ``Ridge`` is ``RidgeClassifier``
  (L2 squared-error loss). They are deliberately distinct so the linear family
  spans none / L1 / L2 / elastic-net without LR↔Ridge redundancy.
* ``FS_ZOO[name](select_k)`` returns a fresh selector transformer (or the string
  ``"passthrough"`` for the no-selection ``All features`` column).
* Custom selectors (Boruta, univariate-LR + BH) fall back to top-k RF importance
  when they would otherwise select too few features.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.special import expit
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    RFE,
    SelectFromModel,
    SelectKBest,
    f_classif,
    mutual_info_classif,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from hcc_multimodal.train.config import RANDOM_STATE

SELECT_K_DEFAULT = 30

# ---------------------------------------------------------------------------
# Model zoo — 10 classifiers matching the reference figure rows.
# ---------------------------------------------------------------------------
MODEL_ZOO: dict[str, tuple[object, dict]] = {
    "LR": (
        LogisticRegression(C=np.inf, solver="lbfgs", max_iter=2000),
        {},
    ),
    "LASSO": (
        LogisticRegression(l1_ratio=1.0, solver="saga", max_iter=5000, random_state=RANDOM_STATE),
        {"model__C": [0.01, 0.1, 1.0, 10.0]},
    ),
    "Ridge": (
        RidgeClassifier(random_state=RANDOM_STATE),
        {"model__alpha": [0.1, 1.0, 10.0, 100.0]},
    ),
    "Elastic Net": (
        LogisticRegression(
            l1_ratio=0.5, solver="saga", max_iter=5000, random_state=RANDOM_STATE,
        ),
        {"model__C": [0.1, 1.0, 10.0], "model__l1_ratio": [0.2, 0.5, 0.8]},
    ),
    "L-SVM": (
        SVC(kernel="linear", probability=True, random_state=RANDOM_STATE),
        {"model__C": [0.01, 0.1, 1.0, 10.0]},
    ),
    "KNN": (
        KNeighborsClassifier(),
        {"model__n_neighbors": [3, 5, 7, 11]},
    ),
    "RF": (
        RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=1),
        {"model__max_depth": [2, 3, 5, None], "model__min_samples_leaf": [1, 5, 10]},
    ),
    "XGB": (
        XGBClassifier(
            random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=1, verbosity=0,
        ),
        {
            "model__n_estimators": [50, 100],
            "model__max_depth": [2, 3],
            "model__learning_rate": [0.05, 0.1],
        },
    ),
    "NNET": (
        MLPClassifier(max_iter=2000, random_state=RANDOM_STATE),
        {"model__hidden_layer_sizes": [(32,), (64,), (32, 16)], "model__alpha": [1e-3, 1e-2]},
    ),
    "NB": (
        GaussianNB(),
        {"model__var_smoothing": [1e-9, 1e-7, 1e-5]},
    ),
}


# ---------------------------------------------------------------------------
# Custom selectors.
# ---------------------------------------------------------------------------
def _top_k_rf_importance(X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """Indices of the top-k features by RF impurity importance (fallback)."""
    rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=1)
    rf.fit(X, y)
    return np.argsort(rf.feature_importances_)[::-1][:k]


def _abs_corr_scores(X, y, corr_fn) -> np.ndarray:
    """Per-feature |correlation| with the label (module-level, picklable helpers)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    scores = np.empty(X.shape[1])
    for j in range(X.shape[1]):
        col = X[:, j]
        if np.std(col) == 0:
            scores[j] = 0.0
            continue
        r = corr_fn(col, y)
        scores[j] = abs(r) if np.isfinite(r) else 0.0
    return scores


def _pearson_score(X, y):
    return _abs_corr_scores(X, y, lambda a, b: stats.pearsonr(a, b)[0])


def _spearman_score(X, y):
    return _abs_corr_scores(X, y, lambda a, b: stats.spearmanr(a, b)[0])


def _kendall_score(X, y):
    return _abs_corr_scores(X, y, lambda a, b: stats.kendalltau(a, b)[0])


def _mutual_info_score_func(X, y):
    return mutual_info_classif(X, y, random_state=RANDOM_STATE)


def _variance_score_func(X, y):
    return np.nan_to_num(np.var(np.asarray(X, dtype=float), axis=0))


class UnivariateLRBHSelector(BaseEstimator, TransformerMixin):
    """Per-feature univariate logistic regression → BH-FDR selection.

    Significance uses a likelihood-ratio test (LRT) of each single-feature logistic
    model vs. the intercept-only model (chi-square, 1 df). Features with BH-adjusted
    q < ``alpha`` are kept; if none pass, falls back to the top-``k`` by LRT statistic.
    """

    def __init__(self, k: int = SELECT_K_DEFAULT, alpha: float = 0.05):
        self.k = k
        self.alpha = alpha

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        p = np.clip(y.mean(), 1e-6, 1 - 1e-6)
        ll0 = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
        stats_arr = np.zeros(X.shape[1])
        pvals = np.ones(X.shape[1])
        for j in range(X.shape[1]):
            col = X[:, j : j + 1]
            if np.std(col) == 0:
                continue
            lr = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000)
            try:
                lr.fit(col, y)
                proba = np.clip(lr.predict_proba(col)[:, 1], 1e-6, 1 - 1e-6)
                ll1 = np.sum(y * np.log(proba) + (1 - y) * np.log(1 - proba))
                lr_stat = 2 * (ll1 - ll0)
                stats_arr[j] = max(lr_stat, 0.0)
                pvals[j] = stats.chi2.sf(stats_arr[j], df=1)
            except Exception:
                continue
        # Benjamini-Hochberg
        order = np.argsort(pvals)
        m = len(pvals)
        ranks = np.arange(1, m + 1)
        qvals = np.empty(m)
        qvals[order] = np.minimum.accumulate((pvals[order] * m / ranks)[::-1])[::-1]
        selected = np.where(qvals < self.alpha)[0]
        if len(selected) == 0:
            selected = np.argsort(stats_arr)[::-1][: self.k]
        self.support_ = np.zeros(X.shape[1], dtype=bool)
        self.support_[selected] = True
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.support_]

    def get_support(self, indices: bool = False):
        return np.where(self.support_)[0] if indices else self.support_


class BorutaSelector(BaseEstimator, TransformerMixin):
    """Boruta all-relevant selection, falling back to top-k RF importance."""

    def __init__(self, k: int = SELECT_K_DEFAULT, max_iter: int = 50):
        self.k = k
        self.max_iter = max_iter

    def fit(self, X, y):
        from boruta import BorutaPy

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=5, random_state=RANDOM_STATE, n_jobs=1
        )
        boruta = BorutaPy(
            rf, n_estimators="auto", max_iter=self.max_iter,
            random_state=RANDOM_STATE, verbose=0,
        )
        support = np.zeros(X.shape[1], dtype=bool)
        try:
            boruta.fit(X, y)
            support = boruta.support_.copy()
            # also accept tentative to avoid empty selections
            if support.sum() == 0 and getattr(boruta, "support_weak_", None) is not None:
                support = boruta.support_weak_.copy()
        except Exception:
            pass
        if support.sum() == 0:
            idx = _top_k_rf_importance(X, y, self.k)
            support[idx] = True
        self.support_ = support
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.support_]

    def get_support(self, indices: bool = False):
        return np.where(self.support_)[0] if indices else self.support_


# ---------------------------------------------------------------------------
# Feature-selection zoo — 12 techniques + a no-selection "All features" column.
# Each entry is a factory ``select_k -> transformer`` (or the "passthrough" str).
# ---------------------------------------------------------------------------
def _l1_selector(k):
    est = LogisticRegression(l1_ratio=1.0, solver="saga", C=1.0, max_iter=5000,
                             random_state=RANDOM_STATE)
    return SelectFromModel(est, max_features=k, threshold=-np.inf)


def _enet_selector(k):
    est = LogisticRegression(l1_ratio=0.5, solver="saga", C=1.0, max_iter=5000,
                             random_state=RANDOM_STATE)
    return SelectFromModel(est, max_features=k, threshold=-np.inf)


def _rf_importance_selector(k):
    est = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=1)
    return SelectFromModel(est, max_features=k, threshold=-np.inf)


def _rfe_selector(k):
    est = LogisticRegression(l1_ratio=0.0, solver="lbfgs", max_iter=2000)
    return RFE(est, n_features_to_select=k)


FS_ZOO: dict[str, object] = {
    "All features": lambda k: "passthrough",
    "Pearson": lambda k: SelectKBest(_pearson_score, k=k),
    "Spearman": lambda k: SelectKBest(_spearman_score, k=k),
    "Kendall": lambda k: SelectKBest(_kendall_score, k=k),
    "Mutual Info": lambda k: SelectKBest(_mutual_info_score_func, k=k),
    "LASSO": _l1_selector,
    "Elastic Net": _enet_selector,
    "Univ. LR (BH)": lambda k: UnivariateLRBHSelector(k=k),
    "RFE": _rfe_selector,
    "Boruta": lambda k: BorutaSelector(k=k),
    "Variance": lambda k: SelectKBest(_variance_score_func, k=k),
    "ANOVA": lambda k: SelectKBest(f_classif, k=k),
    "RF Import.": _rf_importance_selector,
}

# Canonical display order (rows/cols of the heatmap).
MODEL_ORDER = list(MODEL_ZOO)
FS_ORDER = list(FS_ZOO)

# Pipeline parameter that controls how many features each selector keeps. Used to
# fold the number-of-selected-features into GridSearchCV as a tuned hyperparameter.
# ``None`` = the selector has no k (the no-selection "All features" passthrough).
FS_K_PARAM: dict[str, str | None] = {
    "All features": None,
    "Pearson": "selector__k",
    "Spearman": "selector__k",
    "Kendall": "selector__k",
    "Mutual Info": "selector__k",
    "LASSO": "selector__max_features",
    "Elastic Net": "selector__max_features",
    "Univ. LR (BH)": "selector__k",
    "RFE": "selector__n_features_to_select",
    "Boruta": "selector__k",
    "Variance": "selector__k",
    "ANOVA": "selector__k",
    "RF Import.": "selector__max_features",
}


def select_k_grid(fs_name: str, select_k_values) -> dict:
    """GridSearchCV param entry that tunes this selector's feature count.

    Returns ``{param: [k, ...]}`` (e.g. ``{"selector__k": [43, 85]}``) or ``{}`` for
    the passthrough "All features" column, which has no feature-count knob.
    """
    param = FS_K_PARAM[fs_name]
    if param is None or not select_k_values:
        return {}
    return {param: list(select_k_values)}


# ---------------------------------------------------------------------------
# Pipeline factory + helpers.
# ---------------------------------------------------------------------------
def build_grid_pipeline(fs_name: str, model_name: str, select_k: int, memory=None) -> Pipeline:
    """Imputer → scaler → (feature selector) → classifier.

    ``memory`` (a ``joblib.Memory`` or path) caches the imputer/scaler/selector
    fits so expensive selectors (Boruta, RFE) are computed once per fold and reused
    across models and hyperparameter candidates.
    """
    estimator, _ = MODEL_ZOO[model_name]
    factory = FS_ZOO[fs_name]
    selector = factory(select_k)
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("selector", selector),
            ("model", clone(estimator)),
        ],
        memory=memory,
    )


def param_grid(model_name: str) -> dict:
    return MODEL_ZOO[model_name][1]


def has_proba(model_name: str) -> bool:
    return model_name != "Ridge"


def positive_scores(pipe: Pipeline, X) -> np.ndarray:
    """Positive-class score in [0, 1].

    ``predict_proba[:, 1]`` when available, else ``expit(decision_function)`` — the
    logistic squashing keeps proba-less models (Ridge) on a bounded scale whose 0.5
    threshold matches the classifier's natural decision boundary, and which transfers
    consistently across CV folds and cohorts (unlike a per-set min-max).
    """
    if hasattr(pipe, "predict_proba"):
        return np.asarray(pipe.predict_proba(X))[:, 1]
    return expit(np.asarray(pipe.decision_function(X)))


# Backwards-friendly alias (AUROC is invariant to the monotonic expit transform).
raw_scores = positive_scores
