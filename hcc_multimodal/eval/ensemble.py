"""Mean-probability ensemble of per-embedding benchmark-grid heads.

Given several image-only embeddings, an :class:`EnsembleGrid` fits one
:func:`hcc_multimodal.eval.grid.build_grid_pipeline` per embedding (all sharing the
same tuned hyperparameters), then averages the per-embedding positive-class scores.
This keeps the ensemble a drop-in sklearn classifier: ``GridSearchCV`` tunes the
shared ``model__*`` / ``selector__*`` grid exactly as for a single embedding, and the
survival scorer reads ``predict_proba``/``decision_function`` unchanged.

Embeddings are aligned across model ids on the common SID intersection; each block's
columns are namespaced ``{model_id}::{feature}`` so the same combined ``X`` schema is
reused on resection and every transfer cohort.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.base import BaseEstimator, ClassifierMixin

from hcc_multimodal.eval.grid import build_grid_pipeline, positive_scores
from hcc_multimodal.survival.data import CohortData, load_source_aligned

# Constructor arguments — everything else passed to ``set_params`` is a shared
# sub-pipeline hyperparameter (``model__*`` / ``selector__*``) broadcast to each block.
_INIT_PARAMS = ("fs_name", "model_name", "select_k", "blocks", "memory", "params")


class EnsembleGrid(ClassifierMixin, BaseEstimator):
    """Mean positive-class score over one grid pipeline per embedding block.

    ``blocks`` is a list of column-name lists (one per embedding). All sub-pipelines
    share the ``params`` hyperparameters, so ``GridSearchCV`` over the standard grid
    tunes every block in lockstep.
    """

    def __init__(self, fs_name, model_name, select_k, blocks, memory=None, params=None):
        self.fs_name = fs_name
        self.model_name = model_name
        self.select_k = select_k
        self.blocks = blocks
        self.memory = memory
        self.params = params

    # -- sklearn plumbing: keep get_params to the constructor args (so ``clone``
    #    round-trips) and route grid keys (model__*, selector__*) into ``params``.
    def get_params(self, deep=True):
        return {k: getattr(self, k) for k in _INIT_PARAMS}

    def set_params(self, **kw):
        tune = {}
        for k, v in kw.items():
            if k in _INIT_PARAMS:
                setattr(self, k, v)
            else:
                tune[k] = v
        if tune:
            merged = dict(self.params or {})
            merged.update(tune)
            self.params = merged
        return self

    def _make_pipe(self):
        pipe = build_grid_pipeline(self.fs_name, self.model_name, self.select_k, memory=self.memory)
        if self.params:
            pipe.set_params(**self.params)
        return pipe

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.pipes_ = []
        for cols in self.blocks:
            pipe = self._make_pipe()
            pipe.fit(X[cols], y)
            self.pipes_.append(pipe)
        return self

    def _mean_scores(self, X):
        per_block = [positive_scores(p, X[cols]) for p, cols in zip(self.pipes_, self.blocks)]
        return np.mean(per_block, axis=0)

    def predict_proba(self, X):
        s = self._mean_scores(X)
        return np.column_stack([1.0 - s, s])

    def decision_function(self, X):
        s = np.clip(self._mean_scores(X), 1e-9, 1 - 1e-9)
        return logit(s)

    def predict(self, X):
        return (self._mean_scores(X) >= 0.5).astype(int)


def load_ensemble_aligned(model_ids, cohort, **endpoint) -> tuple[CohortData, list[list[str]]]:
    """Combined (patient × concatenated-embedding) frame + per-embedding column blocks.

    Patients are the SID intersection across all ``model_ids`` for this cohort; survival
    outcomes are identical across embeddings, so they are taken from the first. Returns a
    :class:`CohortData` whose ``X`` columns are ``{model_id}::{feature}`` and the list of
    column blocks (one per embedding) for :class:`EnsembleGrid`.
    """
    cds = [load_source_aligned(mid, cohort, **endpoint) for mid in model_ids]
    common = cds[0].X.index
    for cd in cds[1:]:
        common = common.intersection(cd.X.index)
    common = common.sort_values()

    blocks, frames = [], []
    for mid, cd in zip(model_ids, cds):
        Xi = cd.X.loc[common].copy()
        Xi.columns = [f"{mid}::{c}" for c in Xi.columns]
        blocks.append(list(Xi.columns))
        frames.append(Xi)
    X = pd.concat(frames, axis=1)

    base = cds[0]
    return (
        CohortData(cohort, X, base.time.loc[common], base.event.loc[common],
                   base.rfs_2year.loc[common]),
        blocks,
    )
