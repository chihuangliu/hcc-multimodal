"""Dataset for MRI-gene contrastive learning."""

from importlib.resources import files
from pathlib import Path
from enum import StrEnum
from typing import Callable

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import hcc_multimodal
from hcc_multimodal.baselines.data import add_rfs_columns
from hcc_multimodal.baselines.transforms import CPMTransformer
from hcc_multimodal.contrastive.config import GENE_SET
from hcc_multimodal.contrastive.transform import resample_to_spacing, resampled_shape
from hcc_multimodal.utils.data import CLINICAL_CSV, MRI_ARTERIAL_ROOT, RNA_SEQ_CSV

_DATA_ROOT = Path(files(hcc_multimodal).joinpath("")).parent / "data"
_RNA_SEQ_PATH = RNA_SEQ_CSV
_CLINICAL_PATH = CLINICAL_CSV
_MRI_ROOT_PREPROCESSED = MRI_ARTERIAL_ROOT
_MRI_ROOT_RAW = _DATA_ROOT / "Resection" / "Images" / "Resections_with_rna"
_MRI_ROOT_RAW_CACHE = _DATA_ROOT / "mri_resampled"

class MRIType(StrEnum):
    PREPROCESSED = "preprocessed"
    RAW = "raw"
    RAW_BBOX = "raw_bbox"


def _load_gene_matrix(genes: set[str], sort_genes: bool = False) -> pd.DataFrame:
    """Return the (patients x genes) CPM matrix restricted to *genes*.

    The column order is the GeneEncoder's input-slot order, so it must be
    deterministic: `genes` is a set, and iterating it directly gives a
    process-dependent order (string hash randomisation), which makes a trained
    encoder's input-slot -> gene mapping unrecoverable. Both branches below
    avoid that — by default the order is taken from the RNA-seq CSV's own row
    order, which is fixed by the file; `sort_genes=True` sorts alphabetically.
    """
    df = pd.read_csv(_RNA_SEQ_PATH, index_col="Gene Symbol").drop(columns=["ENSEMBL Gene ID"])
    if sort_genes:
        available = sorted(g for g in genes if g in df.index)
    else:
        # dict.fromkeys keeps first-seen order; the CSV has duplicated gene
        # symbols, so plain iteration could list one twice.
        available = list(dict.fromkeys(g for g in df.index if g in genes))
    df = df.loc[available].T                    # (patients, genes)
    df.index = df.index.astype(int)
    values = CPMTransformer().fit_transform(df.values)
    return pd.DataFrame(values, index=df.index, columns=df.columns)


def _load_outcomes(outcome_col: str) -> pd.Series:
    clinical = add_rfs_columns(
        pd.read_csv(_CLINICAL_PATH)
    )
    clinical = clinical.dropna(subset=["SID"])
    clinical["SID"] = clinical["SID"].astype(int)
    return clinical.set_index("SID")[outcome_col].dropna().astype(int)


def _sample_indices(depth: int, n: int | None) -> list[int]:
    if n is None:
        return list(range(depth))
    return np.linspace(0, depth - 1, n, dtype=int).tolist()


def _seg_path(patient_id: int) -> Path:
    """Return the primary (non-multi) segmentation mask for *patient_id*."""
    patient_dir = _MRI_ROOT_RAW / str(patient_id)
    for name in ("hcc_seg_reg_pt.nii.gz", "hcc_seg_reg_mv.nii.gz", "hcc_seg_reg_mv_flag.nii.gz"):
        p = patient_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"No segmentation file found for patient {patient_id}")


def _compute_bbox(
    seg_vol: np.ndarray, pad: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (lo, hi) padded bounding box indices of the non-zero region in *seg_vol*."""
    nz = np.argwhere(seg_vol > 0.5)
    lo = np.maximum(nz.min(axis=0) - pad, 0)
    hi = np.minimum(nz.max(axis=0) + pad, np.array(seg_vol.shape) - 1)
    return lo, hi


def _normalize_slice(s: np.ndarray) -> np.ndarray:
    """Clip at 99th percentile and rescale to [0, 1]."""
    p99 = np.percentile(s, 99)
    s = np.clip(s, 0, p99)
    if p99 > 0:
        s = s / p99
    return s.astype(np.float32)


class MRIGeneDataset(Dataset):
    """Dataset of (MRI slice, gene vector, outcome, patient_id) tuples.

    Each item is one 2D slice from one patient's MRI. Slices are sampled
    uniformly from all three axes (axial, coronal, sagittal).

    Args:
        patient_ids: Patient IDs to include
        gene_matrix: Pre-loaded (patients × genes) DataFrame, CPM-normalised.
            Its column order is the gene vector's slot order and is fixed for
            the dataset's lifetime — `__getitem__` only does `.loc[pid].values`,
            and DataLoader workers share this same object rather than reloading
            the CSV, so every worker and epoch sees the same slot assignment.
        outcomes: Pre-loaded Series indexed by patient ID
        n_per_axis: Slices sampled per axis per patient
        axes: Which axes to sample from — 0=sagittal, 1=coronal, 2=axial.
            Defaults to all three.
        img_size: Output spatial size (square) for each slice
        transform: Transforms applied after resize + 3-channel conversion.
            If None, no normalisation is applied — caller is responsible for
            providing appropriate normalisation for their backbone.
        mri_type: "preprocessed" uses Radiomics/arterial (intensity-normalised,
            already at 1×1×3 mm); "raw" uses Resections_with_rna and resamples
            to 1×1×3 mm on load; "raw_bbox" crops the resampled volume to the
            tumour bounding box (from hcc_seg_reg*.nii.gz) before slicing.
        bbox_pad: Voxel padding added to each face of the tumour bounding box
            (raw_bbox mode only). Default 10.
    """

    def __init__(
        self,
        patient_ids: list[int],
        gene_matrix: pd.DataFrame,
        outcomes: pd.Series,
        n_per_axis: int | None = 10,
        axes: list[int] | None = None,
        img_size: int = 224,
        transform: Callable | None = None,
        mri_type: MRIType = MRIType.PREPROCESSED,
        bbox_pad: int = 10,
    ):
        self.gene_matrix = gene_matrix
        self.outcomes = outcomes
        self.resize = transforms.Resize((img_size, img_size), antialias=True)
        self.transform = transform
        self.mri_type = MRIType(mri_type)
        self.bbox_pad = bbox_pad

        _axes = axes if axes is not None else [0, 1, 2]

        # Pre-computed bounding boxes for RAW_BBOX mode (pid → (lo, hi) arrays).
        self._bboxes: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        # Build flat index: (patient_id, axis, slice_idx)
        self._index: list[tuple[int, int, int]] = []
        for pid in sorted(set(patient_ids)):
            path = _mri_path(pid, mri_type)
            img = nib.load(path)   # header only, no data loaded
            needs_resample = self.mri_type in (MRIType.RAW, MRIType.RAW_BBOX) and not str(path).startswith(str(_MRI_ROOT_RAW_CACHE))

            if self.mri_type == MRIType.RAW_BBOX:
                # Resample the seg mask to determine the tumour bbox.
                seg = nib.load(_seg_path(pid))
                seg_vol = resample_to_spacing(seg)
                lo, hi = _compute_bbox(seg_vol, bbox_pad)
                self._bboxes[pid] = (lo, hi)
                shape = tuple((hi - lo + 1).tolist())
            else:
                shape = resampled_shape(img) if needs_resample else img.shape

            for axis in _axes:
                for i in _sample_indices(shape[axis], n_per_axis):
                    self._index.append((pid, axis, i))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        pid, axis, slice_idx = self._index[i]

        path = _mri_path(pid, self.mri_type)
        img = nib.load(path)
        needs_resample = self.mri_type in (MRIType.RAW, MRIType.RAW_BBOX) and not str(path).startswith(str(_MRI_ROOT_RAW_CACHE))
        vol = resample_to_spacing(img) if needs_resample else np.array(img.dataobj)

        if self.mri_type == MRIType.RAW_BBOX:
            lo, hi = self._bboxes[pid]
            hi_safe = np.minimum(hi, np.array(vol.shape) - 1)
            vol = vol[lo[0]:hi_safe[0]+1, lo[1]:hi_safe[1]+1, lo[2]:hi_safe[2]+1]
            slice_idx = min(slice_idx, vol.shape[axis] - 1)

        s = np.take(vol, slice_idx, axis=axis)       # (H, W) regardless of axis

        s_tensor = torch.from_numpy(_normalize_slice(s)).unsqueeze(0)   # (1, H, W)
        s_tensor = self.resize(s_tensor).repeat(3, 1, 1)                # (3, img_size, img_size)
        if self.transform is not None:
            s_tensor = self.transform(s_tensor)

        gene_vec = torch.tensor(self.gene_matrix.loc[pid].values, dtype=torch.float32)
        return s_tensor, gene_vec, int(self.outcomes[pid]), pid


def _mri_path(patient_id: int, mri_type: MRIType = MRIType.PREPROCESSED) -> Path:
    if mri_type == MRIType.PREPROCESSED:
        return _MRI_ROOT_PREPROCESSED / str(patient_id) / f"{patient_id}.nii.gz"
    # RAW and RAW_BBOX both use the raw arterial volume (cached if available).
    cached = _MRI_ROOT_RAW_CACHE / str(patient_id) / "MRI_liver_arterial.nii.gz"
    if cached.exists():
        return cached
    return _MRI_ROOT_RAW / str(patient_id) / "MRI_liver_arterial.nii.gz"


def build_dataset(
    n_per_axis: int | None = 10,
    axes: list[int] | None = None,
    outcome_col: str = "rfs_2year",
    img_size: int = 224,
    transform: Callable | None = None,
    genes: set[str] | None = None,
    sort_genes: bool = False,
    mri_type: MRIType = MRIType.PREPROCESSED,
    bbox_pad: int = 10,
) -> MRIGeneDataset:
    """Build dataset from patients with MRI, RNA-seq, and a valid outcome.

    Intersects patients available across all three data sources and returns
    a ready-to-use MRIGeneDataset.

    Args:
        n_per_axis: Slices sampled per axis per patient
        axes: Which axes to sample from (0=sagittal, 1=coronal, 2=axial).
            Defaults to all three, giving 3 × n_per_axis slices per patient.
        outcome_col: 'rfs_1year' or 'rfs_2year'
        img_size: Spatial size slices are resized to
        transform: Normalisation/augmentation to apply after resize + 3-channel
            conversion. None means no normalisation.
        genes: Gene set to use; defaults to GENE_SET from config
        sort_genes: If False (default), gene columns follow the RNA-seq CSV's
            row order. If True, they are ordered alphabetically. Either way the
            order is deterministic across processes; see `_load_gene_matrix`.
        mri_type: "preprocessed" (Radiomics/arterial), "raw"
            (Resections_with_rna, resampled to 1×1×3 mm on load), or
            "raw_bbox" (raw, then cropped to the tumour bounding box)
        bbox_pad: Voxel padding around the tumour bounding box
            (raw_bbox mode only). Default 10.

    Returns:
        MRIGeneDataset over the valid patient intersection
    """
    genes = genes or GENE_SET
    gene_matrix = _load_gene_matrix(genes, sort_genes=sort_genes)
    outcomes = _load_outcomes(outcome_col)

    mri_root = _MRI_ROOT_PREPROCESSED if mri_type == MRIType.PREPROCESSED else _MRI_ROOT_RAW
    mri_patients = {int(p.name) for p in mri_root.iterdir() if p.is_dir()}
    valid = sorted(mri_patients & set(gene_matrix.index) & set(outcomes.index))

    return MRIGeneDataset(
        patient_ids=valid,
        gene_matrix=gene_matrix,
        outcomes=outcomes,
        n_per_axis=n_per_axis,
        axes=axes,
        img_size=img_size,
        transform=transform,
        mri_type=mri_type,
        bbox_pad=bbox_pad,
    )
