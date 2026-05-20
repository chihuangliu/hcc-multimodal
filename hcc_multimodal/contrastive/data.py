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

_DATA_ROOT = Path(files(hcc_multimodal).joinpath("")).parent / "data"
_RNA_SEQ_PATH = _DATA_ROOT / "RNA_seq" / "Matrix_output_radiology_only.csv"
_CLINICAL_PATH = _DATA_ROOT / "Clinical" / "2025_Nov_18_ICL_Resection_Clinical_Outcome_soramic_format.csv"
_MRI_ROOT_PREPROCESSED = _DATA_ROOT / "Resection" / "Images" / "Radiomics" / "arterial"
_MRI_ROOT_RAW = _DATA_ROOT / "Resection" / "Images" / "Resections_with_rna"
_MRI_ROOT_RAW_CACHE = _DATA_ROOT / "mri_resampled"

class MRIType(StrEnum):
    PREPROCESSED = "preprocessed"
    RAW = "raw"


def _load_gene_matrix(genes: set[str]) -> pd.DataFrame:
    df = pd.read_csv(_RNA_SEQ_PATH, index_col="Gene Symbol").drop(columns=["ENSEMBL Gene ID"])
    available = [g for g in genes if g in df.index]
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


def _sample_indices(depth: int, n: int) -> list[int]:
    return np.linspace(0, depth - 1, n, dtype=int).tolist()


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
        gene_matrix: Pre-loaded (patients × genes) DataFrame, CPM-normalised
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
            to 1×1×3 mm on load.
    """

    def __init__(
        self,
        patient_ids: list[int],
        gene_matrix: pd.DataFrame,
        outcomes: pd.Series,
        n_per_axis: int = 10,
        axes: list[int] | None = None,
        img_size: int = 224,
        transform: Callable | None = None,
        mri_type: MRIType = MRIType.PREPROCESSED,
    ):
        self.gene_matrix = gene_matrix
        self.outcomes = outcomes
        self.resize = transforms.Resize((img_size, img_size), antialias=True)
        self.transform = transform
        self.mri_type = MRIType(mri_type)

        _axes = axes if axes is not None else [0, 1, 2]

        # Build flat index: (patient_id, axis, slice_idx)
        self._index: list[tuple[int, int, int]] = []
        for pid in sorted(set(patient_ids)):
            path = _mri_path(pid, mri_type)
            img = nib.load(path)   # header only, no data loaded
            needs_resample = mri_type == MRIType.RAW and not str(path).startswith(str(_MRI_ROOT_RAW_CACHE))
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
        needs_resample = self.mri_type == MRIType.RAW and not str(path).startswith(str(_MRI_ROOT_RAW_CACHE))
        vol = resample_to_spacing(img) if needs_resample else np.array(img.dataobj)
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
    cached = _MRI_ROOT_RAW_CACHE / str(patient_id) / "MRI_liver_arterial.nii.gz"
    if cached.exists():
        return cached
    return _MRI_ROOT_RAW / str(patient_id) / "MRI_liver_arterial.nii.gz"


def build_dataset(
    n_per_axis: int = 10,
    axes: list[int] | None = None,
    outcome_col: str = "rfs_2year",
    img_size: int = 224,
    transform: Callable | None = None,
    genes: set[str] | None = None,
    mri_type: MRIType = MRIType.PREPROCESSED,
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
        mri_type: "preprocessed" (Radiomics/arterial) or "raw"
            (Resections_with_rna, resampled to 1×1×3 mm on load)

    Returns:
        MRIGeneDataset over the valid patient intersection
    """
    genes = genes or GENE_SET
    gene_matrix = _load_gene_matrix(genes)
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
    )
