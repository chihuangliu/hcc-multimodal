"""Shared loaders and reference constants for the diagnose scripts.

Thin layer over ``hcc_multimodal.eval.embedding_drift`` so the cohort→parquet
path logic and the KS routine live in exactly one place.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hcc_multimodal.eval.data import load_ablation_outcomes, load_resection_outcomes
from hcc_multimodal.eval.embedding_drift import (
    MODEL_CONFIGS,
    _emb_path,
    ks_drift,  # re-exported for callers that want the summary form
)

__all__ = [
    "COHORTS",
    "SORAMIC_AUROC",
    "LAUSANNE_AUROC",
    "ks_drift",
    "model_suffix",
    "load_embeddings",
    "labeled_sids",
    "perdim_ks_d",
]

COHORTS = ("resection", "soramic", "lausanne")

# 2-year RFS outcome target; used only when labeled=True to keep the geometry on
# the exact patients the transfer AUCs are computed on.
_TARGET = "rfs_2year"

# input suffix (raw / bbox) per model, taken from embedding_drift.MODEL_CONFIGS
_SUFFIX: dict[str, str] = dict(MODEL_CONFIGS)

# Best-head transfer AUROC — reports/0713/0713_embedding_grid_eval_v2.md §5.
# Used only by `ks --corr` to reproduce the drift↔transfer correlation (§6.2).
SORAMIC_AUROC: dict[str, float] = {
    "a6f970d6": 0.494, "dc7e1d10": 0.718, "982a6fa2": 0.606, "a64b245f": 0.684,
    "92b9afed": 0.577, "1361bef2": 0.522, "5e3f71a0": 0.635, "06c598c0": 0.702,
    "12e4ba6a": 0.670, "5d04e6ba": 0.516, "050d401d": 0.669, "8715461c": 0.534,
    "e12b0592": 0.517, "6a1a1bdf": 0.615, "9109a6c2": 0.732, "34e6806f": 0.574,
    "f8aabb75": 0.539,
}
LAUSANNE_AUROC: dict[str, float] = {
    "a6f970d6": 0.618, "dc7e1d10": 0.453, "982a6fa2": 0.600, "a64b245f": 0.556,
    "92b9afed": 0.614, "1361bef2": 0.771, "5e3f71a0": 0.534, "06c598c0": 0.515,
    "12e4ba6a": 0.477, "5d04e6ba": 0.655, "050d401d": 0.544, "8715461c": 0.494,
    "e12b0592": 0.595, "6a1a1bdf": 0.497, "9109a6c2": 0.563, "34e6806f": 0.420,
    "f8aabb75": 0.515,
}


def model_suffix(model_id: str) -> str:
    """raw/bbox input suffix for a model, from embedding_drift.MODEL_CONFIGS."""
    try:
        return _SUFFIX[model_id]
    except KeyError:
        known = ", ".join(sorted(_SUFFIX))
        raise SystemExit(f"unknown model_id {model_id!r}; known: {known}")


def labeled_sids(cohort: str) -> pd.Index:
    """SIDs with a non-null 2-year RFS label for a cohort (the transfer-AUC population)."""
    if cohort == "resection":
        return load_resection_outcomes(_TARGET).index
    # ablation loader uses the internal "lusanne" spelling
    ablation_set = "lusanne" if cohort == "lausanne" else cohort
    return load_ablation_outcomes(ablation_set, _TARGET).index


def load_embeddings(
    model_id: str, cohort: str, suffix: str | None = None, labeled: bool = False
) -> np.ndarray:
    """Patient-level image embeddings for one (model, cohort) as an (n, dim) array.

    labeled=True restricts to SIDs carrying a 2-year RFS label (intersection of the
    embedding cache with the outcome table), matching the transfer-AUC population.
    """
    suffix = suffix or model_suffix(model_id)
    df = pd.read_parquet(_emb_path(model_id, suffix, cohort))
    if labeled:
        df = df.loc[df.index.intersection(labeled_sids(cohort))]
    return df.values.astype(float)


def perdim_ks_d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-dimension KS D-statistic between two cohorts — the raw vector behind
    ``embedding_drift.ks_drift``'s median/mean summary."""
    from scipy.stats import ks_2samp

    return np.array([ks_2samp(a[:, i], b[:, i]).statistic for i in range(a.shape[1])])
