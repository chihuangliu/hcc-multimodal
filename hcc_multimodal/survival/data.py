"""Survival data loaders: time-to-event outcomes + patient-level embeddings.

Reuses the existing eval loaders for clinical file locations / readers and the
shared RFS label derivation, but exposes the *raw* time-to-event signal
(``RFS_central`` months + ``RFS_central_event``) instead of the binary label.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd

from hcc_multimodal.baselines.data import add_rfs_columns
from hcc_multimodal.eval.data import (
    RESECTION_CLINICAL_CSV,
    get_ablation_config,
    load_ablation_radiomics,
    load_resection_radiomics,
    load_resection_radiomics_raw,
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
    # bbox frozen n=all family completed 2026-07-21 (submit_bbox_frozen_train.sh):
    "16acfdd9": "bbox",  # λ=0.0, slice
    "3baefc68": "bbox",  # λ=0.0, patient
    "8fcb5dd3": "bbox",  # λ=0.1, patient
    # 92b9afed continued for 5 more epochs (2026-07-28), bringing the bbox λ=0.1
    # slice config to the 10 epochs the rest of the frozen n=all family was trained for.
    "a5fcd80b": "bbox",  # λ=0.1, slice — continuation of 92b9afed
}


def _input_type(ref) -> str:
    """Return ``"raw"`` or ``"bbox"`` for a model reference's embedding cache.

    ``MODEL_INPUT`` pins the runs predating the ``mri_type`` metadata field. For
    anything newer the run's own ``metadata.json`` is authoritative, so a new run
    needs no dict entry — and, more importantly, a ``raw_bbox`` run no longer
    silently falls back to ``"raw"`` and loads another encoder's cache.
    """
    if ref.run_id in MODEL_INPUT:
        return MODEL_INPUT[ref.run_id]
    meta_path = ref.metadata_path
    if meta_path.exists():
        return "bbox" if json.loads(meta_path.read_text()).get("mri_type") == "raw_bbox" else "raw"
    return "raw"


def _emb_stem(cohort: str, input_type: str) -> str:
    if cohort == "resection":
        return "resection_img_emb"
    return f"ablation_{cohort}_img_emb_{input_type}"


def _read_clinical(cohort: str) -> pd.DataFrame:
    if cohort == "resection":
        return pd.read_csv(RESECTION_CLINICAL_CSV).dropna(how="all")
    cfg = get_ablation_config(cohort)
    return cfg.read_clinical(cfg.clinical_path).dropna(how="all")


def load_survival_outcomes(
    cohort: str,
    tolerance_months: int = 0,
    time_col: str = TIME_COL,
    event_col: str = EVENT_COL,
) -> pd.DataFrame:
    """Return per-patient survival outcomes indexed by integer SID.

    Columns: ``time`` (months, =``time_col``), ``event`` (1=event, 0=censored),
    ``rfs_2year`` (binary 2-year RFS label used to stratify Route-A cross-validation
    and to freeze cutoffs; always derived from RFS_central regardless of the
    time-to-event endpoint). Rows with missing time or event are dropped.

    The default endpoint is recurrence-free survival (``RFS_central``); pass
    ``time_col="TTR_central"`` / ``event_col="TTR_central_event"`` for time-to-recurrence.
    """
    clinical = _read_clinical(cohort)
    clinical = add_rfs_columns(clinical, tolerance_months=tolerance_months)
    clinical = clinical.copy()
    clinical["SID"] = clinical["SID"].astype(int)
    out = clinical.set_index("SID")[[time_col, event_col, "rfs_2year"]].rename(
        columns={time_col: "time", event_col: "event"}
    )
    out = out.dropna(subset=["time", "event"])
    out["event"] = out["event"].astype(int)
    out["time"] = out["time"].astype(float)
    return out


def load_embeddings(model_id: str, cohort: str) -> pd.DataFrame:
    """Load patient-level (mean-pooled) embeddings indexed by integer SID.

    ``model_id`` is a model reference: a bare run id, or ``<run_id>@<epoch>`` to
    read the cache extracted from that run's ``epoch_XXX.pt``.
    """
    from hcc_multimodal.utils.model_ref import parse_model_ref

    ref = parse_model_ref(model_id)
    # keyed on the run id, never the qualified spec, so bbox runs still resolve
    input_type = _input_type(ref)
    path = ref.cache_path(_emb_stem(cohort, input_type))
    if not path.exists():
        raise FileNotFoundError(
            f"No cached {cohort} embeddings for {model_id!r} at {path}. "
            f"Run: python -m hcc_multimodal.eval.eval --mode embedding "
            f"--model-id {model_id} --ablation-set <soramic|lusanne> --target rfs_2year"
        )
    emb = pd.read_parquet(path)
    emb.index = emb.index.astype(int)
    emb.index.name = "SID"
    return emb


def load_radiomic_features(cohort: str, index: pd.Index, raw: bool = False) -> pd.DataFrame:
    """Load arterial-phase radiomic features (multi-lesion averaged) by SID.

    ``raw=True`` reads the resection cohort from the per-lesion TSVs instead of the
    pre-z-scored ``radiomic_cluster.csv``, putting it on the same scale as the ablation
    cohorts (which are always raw). Use it for anything that fits on one cohort and
    transfers to another; ``raw=False`` reproduces the historical (scale-mismatched)
    behaviour of the existing radiomic baselines.
    """
    probe = pd.Series(0, index=index)  # loaders only use the index to filter
    if cohort == "resection":
        loader = load_resection_radiomics_raw if raw else load_resection_radiomics
        X, _ = loader(probe)
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


def _align(
    X: pd.DataFrame,
    cohort: str,
    tolerance_months: int,
    time_col: str = TIME_COL,
    event_col: str = EVENT_COL,
) -> CohortData:
    surv = load_survival_outcomes(
        cohort, tolerance_months=tolerance_months, time_col=time_col, event_col=event_col
    )
    common = X.index.intersection(surv.index).sort_values()
    X, surv = X.loc[common], surv.loc[common]
    return CohortData(cohort, X, surv["time"], surv["event"], surv["rfs_2year"])


def load_aligned(
    model_id: str,
    cohort: str,
    tolerance_months: int = 0,
    time_col: str = TIME_COL,
    event_col: str = EVENT_COL,
) -> CohortData:
    """Intersect embeddings and outcomes on SID and return aligned arrays."""
    return _align(
        load_embeddings(model_id, cohort), cohort, tolerance_months, time_col, event_col
    )


def load_source_aligned(
    source: str,
    cohort: str,
    tolerance_months: int = 0,
    time_col: str = TIME_COL,
    event_col: str = EVENT_COL,
) -> CohortData:
    """Aligned features for any risk-score source.

    ``source`` is ``"radiomic"`` (resection features pre-z-scored, as used by the
    historical baselines), ``"radiomic_raw"`` (resection features re-read raw from the
    per-lesion TSVs, so train and test share one scale) or a contrastive model_id.
    """
    if source in ("radiomic", "radiomic_raw"):
        surv = load_survival_outcomes(
            cohort, tolerance_months=tolerance_months, time_col=time_col, event_col=event_col
        )
        X = load_radiomic_features(cohort, surv.index, raw=(source == "radiomic_raw"))
        return _align(X, cohort, tolerance_months, time_col, event_col)
    return load_aligned(source, cohort, tolerance_months, time_col, event_col)
