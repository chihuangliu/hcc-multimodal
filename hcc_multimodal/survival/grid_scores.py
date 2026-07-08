"""Continuous risk scores for arbitrary (feature-selection, classifier) grid heads.

Generalizes :func:`hcc_multimodal.survival.risk_scores.route_a_scores` to any cell
of the benchmark grid (:mod:`hcc_multimodal.eval.grid`). Hyperparameters are chosen
by inner-CV ``GridSearchCV`` on the labelled resection subset (matching how the
transfer eval picks them); the out-of-fold resection score used to freeze the cutoff
is then computed with those best hyperparameters, and the head is refit on all
labelled resection patients to score the ablation cohort.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from hcc_multimodal.eval.grid import (
    SELECT_K_DEFAULT,
    build_grid_pipeline,
    param_grid,
    positive_scores,
)
from hcc_multimodal.train.config import RANDOM_STATE

from .data import CohortData

N_SPLITS = 3


def route_grid_scores(
    fs_name: str,
    model_name: str,
    train: CohortData,
    test_X: pd.DataFrame,
    select_k: int = SELECT_K_DEFAULT,
    inner_folds: int = 3,
) -> tuple[pd.Series, pd.Series, dict]:
    """Return (out-of-fold resection scores, test scores, best_params)."""
    mask = train.rfs_2year.notna()
    X_tr = train.X[mask]
    y_tr = train.rfs_2year[mask].astype(int)

    gs = GridSearchCV(
        build_grid_pipeline(fs_name, model_name, select_k),
        param_grid(model_name),
        cv=StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=RANDOM_STATE),
        scoring="roc_auc",
        refit=True,
        n_jobs=-1,
        error_score="raise",
    )
    gs.fit(X_tr, y_tr)
    best_params = gs.best_params_

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof = pd.Series(index=X_tr.index, dtype=float)
    for tr_idx, va_idx in skf.split(X_tr, y_tr):
        pipe = build_grid_pipeline(fs_name, model_name, select_k)
        pipe.set_params(**best_params)
        pipe.fit(X_tr.iloc[tr_idx], y_tr.iloc[tr_idx])
        oof.iloc[va_idx] = positive_scores(pipe, X_tr.iloc[va_idx])

    pipe = build_grid_pipeline(fs_name, model_name, select_k)
    pipe.set_params(**best_params)
    pipe.fit(X_tr, y_tr)
    test_scores = pd.Series(positive_scores(pipe, test_X), index=test_X.index)
    return oof, test_scores, best_params


def route_grid_scores_ensemble(
    fs_name: str,
    model_name: str,
    train: CohortData,
    test_X: pd.DataFrame,
    blocks: list[list[str]],
    select_k: int = SELECT_K_DEFAULT,
    inner_folds: int = 3,
) -> tuple[pd.Series, pd.Series, dict]:
    """:func:`route_grid_scores` for a mean-probability :class:`EnsembleGrid` head.

    ``train.X`` / ``test_X`` carry the concatenated ``{model_id}::{feature}`` columns;
    ``blocks`` is the per-embedding column grouping. Hyperparameters are tuned once by
    inner-CV over the ensemble; OOF and test scores are its averaged positive-class score.
    """
    from hcc_multimodal.eval.ensemble import EnsembleGrid

    mask = train.rfs_2year.notna()
    X_tr = train.X[mask]
    y_tr = train.rfs_2year[mask].astype(int)

    def _est():
        return EnsembleGrid(fs_name, model_name, select_k, blocks)

    gs = GridSearchCV(
        _est(), param_grid(model_name),
        cv=StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=RANDOM_STATE),
        scoring="roc_auc", refit=True, n_jobs=-1, error_score="raise",
    )
    gs.fit(X_tr, y_tr)
    best_params = gs.best_params_

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof = pd.Series(index=X_tr.index, dtype=float)
    for tr_idx, va_idx in skf.split(X_tr, y_tr):
        est = _est().set_params(**best_params)
        est.fit(X_tr.iloc[tr_idx], y_tr.iloc[tr_idx])
        oof.iloc[va_idx] = positive_scores(est, X_tr.iloc[va_idx])

    est = _est().set_params(**best_params)
    est.fit(X_tr, y_tr)
    test_scores = pd.Series(positive_scores(est, test_X), index=test_X.index)
    return oof, test_scores, best_params
