"""Survival data loaders: time-to-event outcomes + patient-level embeddings.

Reuses the existing eval loaders for clinical file locations / readers and the
shared RFS label derivation, but exposes the *raw* time-to-event signal
(``RFS_central`` months + ``RFS_central_event``) instead of the binary label.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from hcc_multimodal.baselines.data import add_rfs_columns
from hcc_multimodal.eval.data import (
    RESECTION_CLINICAL_CSV,
    TRAINING_ROOT,
    get_ablation_config,
    load_ablation_radiomics,
    load_resection_radiomics,
)

TIME_COL = "RFS_central"
EVENT_COL = "RFS_central_event"

# Per-model contrastive input type (raw MRI vs bbox crop), from the 0608 report.
MODEL_INPUT = {
    "6a1a1bdf": "raw", "1361bef2": "raw", "982a6fa2": "raw", "a6f970d6": "raw",
    "12e4ba6a": "raw", "34e6806f": "raw", "5d04e6ba": "raw", "9109a6c2": "raw",
    "dc7e1d10": "raw", "5e3f71a0": "raw", "a64b245f": "raw", "06c598c0": "raw",
    "050d401d": "bbox", "f8aabb75": "bbox", "e12b0592": "bbox", "8715461c": "bbox",
    "92b9afed": "bbox",
}


def _emb_filename(cohort: str, input_type: str) -> str:
    if cohort == "resection":
        return "resection_img_emb.parquet"
    return f"ablation_{cohort}_img_emb_{input_type}.parquet"


def _read_clinical(cohort: str) -> pd.DataFrame:
    if cohort == "resection":
        return pd.read_csv(RESECTION_CLINICAL_CSV).dropna(how="all")
    cfg = get_ablation_config(cohort)
    return cfg.read_clinical(cfg.clinical_path).dropna(how="all")


def load_survival_outcomes(cohort: str, tolerance_months: int = 0) -> pd.DataFrame:
    """Return per-patient survival outcomes indexed by integer SID.

    Columns: ``time`` (months, =RFS_central), ``event`` (1=recurrence, 0=censored),
    ``rfs_2year`` (binary label used to stratify Route-A cross-validation).
    Rows with missing time or event are dropped.
    """
    clinical = _read_clinical(cohort)
    clinical = add_rfs_columns(clinical, tolerance_months=tolerance_months)
    clinical = clinical.copy()
    clinical["SID"] = clinical["SID"].astype(int)
    out = clinical.set_index("SID")[[TIME_COL, EVENT_COL, "rfs_2year"]].rename(
        columns={TIME_COL: "time", EVENT_COL: "event"}
    )
    out = out.dropna(subset=["time", "event"])
    out["event"] = out["event"].astype(int)
    out["time"] = out["time"].astype(float)
    return out


def load_embeddings(model_id: str, cohort: str) -> pd.DataFrame:
    """Load patient-level (mean-pooled) embeddings indexed by integer SID."""
    input_type = MODEL_INPUT.get(model_id, "raw")
    path = TRAINING_ROOT / model_id / "cached_embeddings" / _emb_filename(cohort, input_type)
    emb = pd.read_parquet(path)
    emb.index = emb.index.astype(int)
    emb.index.name = "SID"
    return emb


def load_radiomic_features(cohort: str, index: pd.Index) -> pd.DataFrame:
    """Load arterial-phase radiomic features (multi-lesion averaged) by SID."""
    probe = pd.Series(0, index=index)  # loaders only use the index to filter
    if cohort == "resection":
        X, _ = load_resection_radiomics(probe)
    else:
        X, _ = load_ablation_radiomics(cohort, probe, "average")
        X.index = X.index.astype(int)
    return X


@dataclass
class CohortData:
    """Aligned embeddings + survival outcomes for one (model, cohort)."""

    cohort: str
    X: pd.DataFrame  # (n_patients, embed_dim)
    time: pd.Series
    event: pd.Series
    rfs_2year: pd.Series  # may contain NaN where 2yr label undefined


def _align(X: pd.DataFrame, cohort: str, tolerance_months: int) -> CohortData:
    surv = load_survival_outcomes(cohort, tolerance_months=tolerance_months)
    common = X.index.intersection(surv.index).sort_values()
    X, surv = X.loc[common], surv.loc[common]
    return CohortData(cohort, X, surv["time"], surv["event"], surv["rfs_2year"])


def load_aligned(model_id: str, cohort: str, tolerance_months: int = 0) -> CohortData:
    """Intersect embeddings and outcomes on SID and return aligned arrays."""
    return _align(load_embeddings(model_id, cohort), cohort, tolerance_months)


def load_source_aligned(source: str, cohort: str, tolerance_months: int = 0) -> CohortData:
    """Aligned features for any risk-score source.

    ``source`` is ``"radiomic"`` or a contrastive model_id.
    """
    if source == "radiomic":
        surv = load_survival_outcomes(cohort, tolerance_months=tolerance_months)
        X = load_radiomic_features(cohort, surv.index)
        return _align(X, cohort, tolerance_months)
    return load_aligned(source, cohort, tolerance_months)
