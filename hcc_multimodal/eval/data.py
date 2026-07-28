"""Data loading utilities for the ablation and resection evaluation pipelines."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from hcc_multimodal.baselines.data import add_rfs_columns
from hcc_multimodal.eval.eval_utils import PROJECT_ROOT
from hcc_multimodal.contrastive.encoders import ImageEncoder
from hcc_multimodal.contrastive.transform import resample_to_spacing
from hcc_multimodal.utils.data import (
    CLINICAL_CSV,
    MRI_ARTERIAL_ROOT,
    RADIOMICS_FEATURES,
    RADIOMIC_CLUSTER_CSV,
    load_resection_arterial_radiomics,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_ROOT = PROJECT_ROOT / "data"
TRAINING_ROOT = PROJECT_ROOT / "training" / "contrastive"

_ABL_ROOT = DATA_ROOT / "Ablation"
_SORAMIC_ROOT = _ABL_ROOT / "soramic"
_LUSANNE_ROOT = _ABL_ROOT / "lusanne"

RESECTION_CLINICAL_CSV = CLINICAL_CSV
RESECTION_RADIOMIC_CSV = RADIOMIC_CLUSTER_CSV
RESECTION_MRI_ROOT = MRI_ARTERIAL_ROOT

CONTRASTIVE_METADATA_FILENAME = "metadata.json"
CONTRASTIVE_CHECKPOINT_FILENAME = "best_model.pt"

RESECTION_EMB_CACHE = "resection_img_emb.parquet"


# ---------------------------------------------------------------------------
# Dataset config
# ---------------------------------------------------------------------------
@dataclass
class AblationDatasetConfig:
    clinical_path: Path
    radiomic_root: Path
    mri_root: Path
    masks_root: Path
    pid_to_mri_relpath: Callable[[int], str]
    pid_to_mask_prefix: Callable[[int], str]
    read_clinical: Callable[[Path], pd.DataFrame]


def get_ablation_config(ablation_set: str) -> AblationDatasetConfig:
    if ablation_set == "soramic":
        return AblationDatasetConfig(
            clinical_path=_SORAMIC_ROOT / "Clinical" / "2025_Feb_21_RFA_Clinical_Data.csv",
            radiomic_root=_SORAMIC_ROOT / "Radiomics" / "arterial_phase",
            mri_root=_SORAMIC_ROOT / "Radiomics_nifti_all" / "RFA_images",
            masks_root=_SORAMIC_ROOT / "Radiomics_nifti_all" / "masks",
            pid_to_mri_relpath=lambda pid: f"{pid}/MRI_dyn_arterial.nii.gz",
            pid_to_mask_prefix=lambda pid: str(pid),
            read_clinical=lambda p: pd.read_csv(p, encoding="latin-1"),
        )
    if ablation_set == "lusanne":
        return AblationDatasetConfig(
            clinical_path=_LUSANNE_ROOT / "Clinical" / "lusanne_clinical_annotation.xlsx",
            radiomic_root=_LUSANNE_ROOT / "Radiomics" / "arterial_phase",
            mri_root=_LUSANNE_ROOT / "Radiomics_nifti_all" / "RFA_images",
            masks_root=_LUSANNE_ROOT / "Radiomics_nifti_all" / "masks",
            pid_to_mri_relpath=lambda pid: f"{pid:04d}/date_one/MRI_liver_arterial.nii.gz",
            pid_to_mask_prefix=lambda pid: f"{pid:03d}",
            read_clinical=lambda p: pd.read_excel(p),
        )
    raise ValueError(f"Unknown ablation_set: {ablation_set!r}. Choose 'soramic' or 'lusanne'.")


# ---------------------------------------------------------------------------
# Outcome loaders
# ---------------------------------------------------------------------------
def load_ablation_outcomes(
    ablation_set: str, target: str, tolerance_months: int = 0
) -> pd.Series:
    cfg = get_ablation_config(ablation_set)
    clinical = cfg.read_clinical(cfg.clinical_path).dropna(how="all")
    clinical = add_rfs_columns(clinical, tolerance_months=tolerance_months)
    clinical["SID"] = clinical["SID"].astype(int)
    return clinical.set_index("SID")[target].dropna().astype(int)


def load_resection_outcomes(target: str, tolerance_months: int = 0) -> pd.Series:
    clinical = pd.read_csv(RESECTION_CLINICAL_CSV).dropna(how="all")
    clinical = add_rfs_columns(clinical, tolerance_months=tolerance_months)
    clinical["SID"] = clinical["SID"].astype(int)
    return clinical.set_index("SID")[target].dropna().astype(int)


# ---------------------------------------------------------------------------
# Radiomic loaders
# ---------------------------------------------------------------------------
def load_ablation_radiomics(
    ablation_set: str,
    outcomes: pd.Series,
    multi_lesion: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load ablation radiomic features from per-lesion TSV files.

    multi_lesion='average'    → one row per patient (features averaged across lesions)
    multi_lesion='per_lesion' → one row per lesion, outcome replicated from patient
    """
    cfg = get_ablation_config(ablation_set)
    rows, lesion_ids = [], []
    for lesion_dir in sorted(cfg.radiomic_root.iterdir()):
        tsv = lesion_dir / f"{lesion_dir.name}.nii-results.tsv"
        if not tsv.exists():
            continue
        df = pd.read_csv(tsv, sep="\t", nrows=1)
        missing = [f for f in RADIOMICS_FEATURES if f not in df.columns]
        if missing:
            continue
        rows.append(df[RADIOMICS_FEATURES].iloc[0].values)
        lesion_ids.append(lesion_dir.name)

    X_raw = pd.DataFrame(rows, index=lesion_ids, columns=RADIOMICS_FEATURES)
    patient_ids = X_raw.index.map(lambda s: int(s.rsplit("_", 1)[0]))
    X_raw["_patient_id"] = patient_ids
    X_raw = X_raw[X_raw["_patient_id"].isin(outcomes.index)]

    if multi_lesion == "average":
        X = X_raw.groupby("_patient_id")[RADIOMICS_FEATURES].mean()
        X.index.name = "SID"
        y = outcomes.loc[X.index]
        return X, y

    X = X_raw.drop(columns=["_patient_id"])
    y = pd.Series(
        [outcomes.loc[int(s.rsplit("_", 1)[0])] for s in X.index],
        index=X.index,
    )
    return X, y


def load_resection_radiomics(outcomes: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Arterial features from ``radiomic_cluster.csv`` — **already z-scored** per feature.

    The CSV was standardised within the 60 resection patients (column means ~1e-16,
    sample std ~1.0084 = sqrt(60/59)), so these features are *not* on the same scale as
    :func:`load_ablation_radiomics`, which returns raw extractor output. Training here
    and transferring to an ablation cohort therefore crosses a many-order-of-magnitude
    scale break; use :func:`load_resection_radiomics_raw` for cross-cohort work.
    """
    df = pd.read_csv(RESECTION_RADIOMIC_CSV).dropna(how="all")
    art_cols = [f"{f}_art" for f in RADIOMICS_FEATURES]
    df = df[["SID"] + art_cols].copy()
    df.columns = ["SID"] + RADIOMICS_FEATURES
    df["SID"] = df["SID"].astype(int)
    df = df.set_index("SID")
    common = df.index.intersection(outcomes.index)
    return df.loc[common], outcomes.loc[common]


def load_resection_radiomics_raw(outcomes: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Raw arterial features from the per-lesion TSVs, matching the ablation cohorts.

    Same extraction and same units as :func:`load_ablation_radiomics` (one lesion per
    resection patient, directory name = ``SID``), so a head fitted here transfers to
    Soramic/Lausanne without a scale break — unlike the pre-z-scored
    :func:`load_resection_radiomics`.
    """
    df = load_resection_arterial_radiomics()
    sid = df["Scan name"].astype(str).str.split(".", n=1).str[0].astype(int)
    X = df[RADIOMICS_FEATURES].copy()
    X.index = pd.Index(sid, name="SID")
    common = X.index.intersection(outcomes.index)
    return X.loc[common], outcomes.loc[common]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _normalize_slice(s: np.ndarray) -> np.ndarray:
    p99 = np.percentile(s, 99)
    s = np.clip(s, 0, p99)
    if p99 > 0:
        s = s / p99
    return s.astype(np.float32)


def _sample_indices(depth: int, n: int | None) -> list[int]:
    if n is None:
        return list(range(depth))
    return np.linspace(0, depth - 1, n, dtype=int).tolist()


def _compute_bbox(seg_vol: np.ndarray, pad: int) -> tuple[np.ndarray, np.ndarray]:
    nz = np.argwhere(seg_vol > 0.5)
    lo = np.maximum(nz.min(axis=0) - pad, 0)
    hi = np.minimum(nz.max(axis=0) + pad, np.array(seg_vol.shape) - 1)
    return lo, hi


# ---------------------------------------------------------------------------
# MRI dataset
# ---------------------------------------------------------------------------
class _MRIDataset(Dataset):
    """Flat (patient_id, slice) index for image embedding extraction.

    Supports two modes:
    - Plain (masks_root=None): optionally resample to 1×1×3 mm, then slice.
    - BBox (masks_root given): crop around the union of all patient-level masks
      from masks_root, then slice.
    """

    def __init__(
        self,
        patient_ids: list[int],
        mri_root: Path,
        pid_to_mri_relpath: Callable[[int], str],
        n_per_axis: int | None,
        axis: int,
        img_size: int,
        vit_transform,
        resample: bool = False,
        masks_root: Path | None = None,
        pid_to_mask_prefix: Callable[[int], str] | None = None,
        bbox_pad: int = 10,
    ):
        self.axis = axis
        self.resize = transforms.Resize((img_size, img_size), antialias=True)
        self.vit_transform = vit_transform
        bbox_mode = masks_root is not None

        self._index: list[tuple[int, int]] = []
        self._vols: dict[int, np.ndarray] = {}

        for pid in patient_ids:
            mri_path = mri_root / pid_to_mri_relpath(pid)
            if not mri_path.exists():
                continue

            if bbox_mode:
                prefix = pid_to_mask_prefix(pid)
                mask_files = sorted(masks_root.glob(f"{prefix}_hcc_seg_reg_*.nii.gz"))
                if not mask_files:
                    continue
                full_vol = resample_to_spacing(nib.load(mri_path))
                seg_arrs = [resample_to_spacing(nib.load(mf)) for mf in mask_files]
                seg_vol = (
                    np.maximum.reduce(seg_arrs) if len(seg_arrs) > 1 else seg_arrs[0]
                )
                if seg_vol.max() < 0.5:
                    continue
                lo, hi = _compute_bbox(seg_vol, bbox_pad)
                hi_safe = np.minimum(hi, np.array(full_vol.shape[:3]) - 1)
                vol = full_vol[lo[0]:hi_safe[0]+1, lo[1]:hi_safe[1]+1, lo[2]:hi_safe[2]+1]
            else:
                img = nib.load(mri_path)
                if resample:
                    vol = resample_to_spacing(img)
                else:
                    vol = np.squeeze(np.array(img.dataobj))
                    if vol.ndim == 4:
                        vol = vol[..., 0]

            self._vols[pid] = vol
            for si in _sample_indices(vol.shape[axis], n_per_axis):
                self._index.append((pid, si))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        pid, si = self._index[i]
        vol = self._vols[pid]
        si = min(si, vol.shape[self.axis] - 1)
        s = np.take(vol, si, axis=self.axis)
        t = torch.from_numpy(_normalize_slice(s)).unsqueeze(0)
        t = self.resize(t).repeat(3, 1, 1)
        return self.vit_transform(t), pid


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------
def extract_image_embeddings(
    img_enc: ImageEncoder,
    patient_ids: list[int],
    mri_root: Path,
    pid_to_mri_relpath: Callable[[int], str],
    meta: dict,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    cache_path: Path | None = None,
    resample: bool = False,
    masks_root: Path | None = None,
    pid_to_mask_prefix: Callable[[int], str] | None = None,
    bbox_pad: int = 10,
    overwrite_cache: bool = False,
) -> pd.DataFrame:
    """Return (n_patients × embed_dim) DataFrame of mean-pooled image embeddings.

    If cache_path exists the file is loaded directly; on a miss the result is
    written to cache_path.
    """
    if cache_path is not None and cache_path.exists() and not overwrite_cache:
        print(f"  Loading cached embeddings from {cache_path}")
        return pd.read_parquet(cache_path)

    from hcc_multimodal.contrastive.encoders import BACKBONE_TRANSFORMS

    vit_transform = BACKBONE_TRANSFORMS[meta["model"]]()

    dataset = _MRIDataset(
        patient_ids=patient_ids,
        mri_root=mri_root,
        pid_to_mri_relpath=pid_to_mri_relpath,
        n_per_axis=meta["n_per_axis"],
        axis=meta["axes"] if isinstance(meta["axes"], int) else meta["axes"][0],
        img_size=meta["img_size"],
        vit_transform=vit_transform,
        resample=resample,
        masks_root=masks_root,
        pid_to_mask_prefix=pid_to_mask_prefix,
        bbox_pad=bbox_pad,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    embed_dim = meta["embed_dim"]
    img_sum: dict[int, np.ndarray] = {}
    img_count: dict[int, int] = {}

    img_enc.eval()
    with torch.no_grad():
        for imgs, pids in loader:
            imgs = imgs.to(device)
            out = img_enc(imgs).cpu().numpy()
            for i, pid in enumerate(pids.tolist()):
                img_sum.setdefault(pid, np.zeros(embed_dim))
                img_count[pid] = img_count.get(pid, 0) + 1
                img_sum[pid] += out[i]

    valid_pids = sorted(img_sum)
    emb = np.stack([img_sum[p] / img_count[p] for p in valid_pids])
    cols = [f"img_{i}" for i in range(embed_dim)]
    df = pd.DataFrame(emb, index=valid_pids, columns=cols)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path)
        print(f"  Cached embeddings → {cache_path}")

    return df


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------
def load_contrastive_model(
    model_id: str, device: torch.device
) -> tuple[ImageEncoder, dict]:
    run_dir = TRAINING_ROOT / model_id
    meta = json.loads((run_dir / CONTRASTIVE_METADATA_FILENAME).read_text())
    img_enc = ImageEncoder(
        meta["model"], meta["embed_dim"], meta["freeze_backbone"]
    ).to(device)
    ckpt = torch.load(
        run_dir / CONTRASTIVE_CHECKPOINT_FILENAME, map_location=device, weights_only=False
    )
    img_enc.load_state_dict(ckpt["img_enc"])
    img_enc.eval()
    return img_enc, meta
