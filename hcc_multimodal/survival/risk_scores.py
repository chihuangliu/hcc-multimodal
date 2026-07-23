"""Continuous risk scores from embeddings, via two routes.

Route A: reuse the 2-year RFS classifier (SelectKBest + LR/RF head) probability.
Route B: penalized Cox PH on PCA-reduced embeddings (partial hazard).

For both routes the *training* (resection) score used to set the cutoff is the
out-of-fold cross-validated score, avoiding the in-sample optimism that a head
refit on all resection patients would carry. The model is then refit on all
resection patients to score the held-out ablation cohorts.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from lifelines import CoxPHFitter
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from hcc_multimodal.eval.eval_utils import DOWNSTREAM_MODELS, build_pipeline
from hcc_multimodal.train.config import RANDOM_STATE, SELECT_K

from .data import CohortData

N_SPLITS = 3


# ---------------------------------------------------------------------------
# Route A — reuse the 2-year RFS classifier probability
# ---------------------------------------------------------------------------
def route_a_scores(
    head: str,
    train: CohortData,
    test_X: pd.DataFrame,
    select_k: int = SELECT_K,
) -> tuple[pd.Series, pd.Series]:
    """Return (out-of-fold resection scores, test scores).

    Only resection patients with a defined ``rfs_2year`` label can train the
    classifier head; the cutoff is the median over that labelled subset.
    """
    mask = train.rfs_2year.notna()
    X_tr = train.X[mask]
    y_tr = train.rfs_2year[mask].astype(int)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof = pd.Series(index=X_tr.index, dtype=float)
    for tr_idx, va_idx in skf.split(X_tr, y_tr):
        pipe = build_pipeline(DOWNSTREAM_MODELS[head], select_k)
        pipe.fit(X_tr.iloc[tr_idx], y_tr.iloc[tr_idx])
        oof.iloc[va_idx] = pipe.predict_proba(X_tr.iloc[va_idx])[:, 1]

    pipe = build_pipeline(DOWNSTREAM_MODELS[head], select_k)
    pipe.fit(X_tr, y_tr)
    test_scores = pd.Series(pipe.predict_proba(test_X)[:, 1], index=test_X.index)
    return oof, test_scores


# ---------------------------------------------------------------------------
# Route B — penalized Cox PH on PCA-reduced embeddings
# ---------------------------------------------------------------------------
@dataclass
class _CoxModel:
    scaler: StandardScaler
    pca: PCA
    cph: CoxPHFitter
    pc_cols: list[str]

    def partial_hazard(self, X: pd.DataFrame) -> pd.Series:
        Z = self.pca.transform(self.scaler.transform(X))
        df = pd.DataFrame(Z, columns=self.pc_cols, index=X.index)
        return self.cph.predict_partial_hazard(df)


def _fit_cox(
    X: pd.DataFrame,
    time: pd.Series,
    event: pd.Series,
    n_components: int,
    penalizer: float,
    l1_ratio: float,
) -> _CoxModel:
    n_comp = min(n_components, X.shape[1], len(X) - 1)
    scaler = StandardScaler().fit(X)
    pca = PCA(n_components=n_comp, random_state=RANDOM_STATE).fit(scaler.transform(X))
    pc_cols = [f"pc_{i}" for i in range(n_comp)]
    df = pd.DataFrame(pca.transform(scaler.transform(X)), columns=pc_cols, index=X.index)
    df["time"] = time.values
    df["event"] = event.values
    cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
    cph.fit(df, duration_col="time", event_col="event")
    return _CoxModel(scaler, pca, cph, pc_cols)


def route_b_scores(
    train: CohortData,
    test_X: pd.DataFrame,
    n_components: int = 10,
    penalizer: float = 0.1,
    l1_ratio: float = 0.0,
) -> tuple[pd.Series, pd.Series]:
    """Return (out-of-fold resection partial hazards, test partial hazards)."""
    X_tr = train.X
    time = train.time
    event = train.event

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof = pd.Series(index=X_tr.index, dtype=float)
    for tr_idx, va_idx in skf.split(X_tr, event):
        model = _fit_cox(
            X_tr.iloc[tr_idx], time.iloc[tr_idx], event.iloc[tr_idx],
            n_components, penalizer, l1_ratio,
        )
        oof.iloc[va_idx] = model.partial_hazard(X_tr.iloc[va_idx]).values

    full = _fit_cox(X_tr, time, event, n_components, penalizer, l1_ratio)
    test_scores = full.partial_hazard(test_X)
    return oof, test_scores
