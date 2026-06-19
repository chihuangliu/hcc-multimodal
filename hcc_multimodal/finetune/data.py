"""Data loading for DINO self-supervised finetuning on MRI."""

from collections import OrderedDict
from importlib.resources import files
from pathlib import Path
from typing import Callable

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import hcc_multimodal
from hcc_multimodal.baselines.data import add_rfs_columns
from hcc_multimodal.contrastive.transform import resample_to_spacing, resampled_shape

_DATA_ROOT = Path(files(hcc_multimodal).joinpath("")).parent / "data"

_RESECTION_NIFTI_ALL = _DATA_ROOT / "Resection" / "Images" / "Resections_nifti_all"
_SORAMIC_MRI_ROOT = _DATA_ROOT / "Ablation" / "soramic" / "Radiomics_nifti_all" / "RFA_images"
_LUSANNE_MRI_ROOT = _DATA_ROOT / "Ablation" / "lusanne" / "Radiomics_nifti_all" / "RFA_images"
_SORAMIC_CLINICAL = _DATA_ROOT / "Ablation" / "soramic" / "Clinical" / "2025_Feb_21_RFA_Clinical_Data.csv"
_LUSANNE_CLINICAL = _DATA_ROOT / "Ablation" / "lusanne" / "Clinical" / "lusanne_clinical_annotation.xlsx"

COHORT_CHOICES = ["resection", "soramic", "lausanne"]
PHASE_CHOICES = ["arterial", "portovenous", "delayed", "hpb"]

# Per-cohort mapping: phase name → relative path inside the patient directory.
# Missing entries mean that phase is not available for that cohort.
_PHASE_FILES: dict[str, dict[str, str]] = {
    "resection": {
        "arterial":    "MRI_liver_arterial.nii.gz",
        "portovenous": "MRI_liver_portovenous.nii.gz",
        "delayed":     "MRI_liver_delayed.nii.gz",
        "hpb":         "MRI_liver_hpb.nii.gz",
    },
    "soramic": {
        "arterial":    "MRI_dyn_arterial.nii.gz",
        "portovenous": "MRI_dyn_portalvenous.nii.gz",
        "delayed":     "MRI_dyn_venous.nii.gz",
        "hpb":         "MRI_postcontrast_hepatobiliary_3D_AX.nii.gz",
    },
    "lausanne": {
        "arterial":    "date_one/MRI_liver_arterial.nii.gz",
        "portovenous": "date_one/MRI_liver_portovenous.nii.gz",
        "delayed":     "date_one/MRI_liver_delayed.nii.gz",
    },
}


def _ablation_patients_with_2yr_outcome(cohort: str) -> set[int]:
    """Return patient IDs from an ablation cohort that have a valid 2yr RFS outcome."""
    if cohort == "soramic":
        clinical = pd.read_csv(_SORAMIC_CLINICAL, encoding="latin-1").dropna(how="all")
    elif cohort == "lausanne":
        clinical = pd.read_excel(_LUSANNE_CLINICAL).dropna(how="all")
    else:
        raise ValueError(cohort)
    clinical = add_rfs_columns(clinical)
    clinical["SID"] = clinical["SID"].astype(int)
    return set(clinical.set_index("SID")["rfs_2year"].dropna().index)


def collect_cohort_patients(
    cohort: str,
    phases: list[str] | None = None,
) -> list[tuple[int, Path]]:
    """Return list of (pid, mri_path) for a cohort.

    Each requested phase that exists on disk produces a separate entry, so one
    patient can appear multiple times (once per phase). All MRI requires
    resampling to 1×1×3 mm.

    For 'soramic' and 'lausanne', patients with a known 2yr RFS outcome are
    excluded (they are reserved as downstream prediction test sets).

    Args:
        cohort: one of 'resection', 'soramic', 'lausanne'
        phases: phase names to include; defaults to ['arterial'].
            Phases not available for a cohort or missing on disk are silently
            skipped. Choose from: 'arterial', 'portovenous', 'delayed', 'hpb'.

    Returns:
        List of (patient_id, mri_path) pairs, one per (patient, phase) that exists.
    """
    if phases is None:
        phases = ["arterial"]

    phase_map = _PHASE_FILES.get(cohort, {})
    # Resolve which phases are defined for this cohort
    phase_relpaths: list[str] = [phase_map[p] for p in phases if p in phase_map]

    entries: list[tuple[int, Path]] = []

    if cohort == "resection":
        for d in sorted(_RESECTION_NIFTI_ALL.iterdir()):
            if not d.is_dir():
                continue
            try:
                pid = int(d.name)
            except ValueError:
                continue
            for relpath in phase_relpaths:
                mri = d / relpath
                if mri.exists():
                    entries.append((pid, mri))

    elif cohort in ("soramic", "lausanne"):
        excluded = _ablation_patients_with_2yr_outcome(cohort)
        mri_root = _SORAMIC_MRI_ROOT if cohort == "soramic" else _LUSANNE_MRI_ROOT

        for d in sorted(mri_root.iterdir()):
            if not d.is_dir():
                continue
            try:
                pid = int(d.name)
            except ValueError:
                continue
            if pid in excluded:
                continue
            for relpath in phase_relpaths:
                mri = d / relpath
                if mri.exists():
                    entries.append((pid, mri))

    else:
        raise ValueError(f"Unknown cohort {cohort!r}. Choose from {COHORT_CHOICES}.")

    return entries


def _normalize_slice(s: np.ndarray) -> np.ndarray:
    """Clip at 99th percentile and rescale to [0, 1]."""
    p99 = np.percentile(s, 99)
    s = np.clip(s, 0, p99)
    if p99 > 0:
        s = s / p99
    return s.astype(np.float32)


def _sample_indices(depth: int, n: int | None) -> list[int]:
    if n is None:
        return list(range(depth))
    return np.linspace(0, depth - 1, n, dtype=int).tolist()


class MultiCropDataset(Dataset):
    """MRI slice dataset returning multi-crop views for DINO finetuning.

    Each item is (views, pid) where views = [global1, global2, local1, ..., localN].
    Global views are the full resized slice; local views use random crops baked into
    local_transform (e.g. v2.RandomCrop → v2.Resize).

    All MRI volumes are resampled to 1×1×3 mm voxel spacing on load.

    Args:
        entries: list of (pid, mri_path) pairs
        n_per_axis: slices sampled per axis per patient (None = all)
        axes: which axes to slice (0=sagittal, 1=coronal, 2=axial)
        img_size: output size for global views; passed to self.resize
        n_local_crops: number of local augmented views per slice
        global_transform: augmentation applied to global (full-slice) views
        local_transform: augmentation applied to local (cropped) views;
            must handle its own cropping (e.g. starts with v2.RandomCrop)
        cache_size: number of resampled volumes to keep in a per-worker LRU
            cache (0 disables). Avoids re-running nib.load + resample_to_spacing
            once per slice, which dominates epoch time when n_per_axis is large.
    """

    def __init__(
        self,
        entries: list[tuple[int, Path]],
        n_per_axis: int | None,
        axes: list[int],
        img_size: int,
        n_local_crops: int,
        global_transform: Callable,
        local_transform: Callable,
        cache_size: int = 64,
    ):
        self.resize = transforms.Resize((img_size, img_size), antialias=True)
        self.n_local_crops = n_local_crops
        self.global_transform = global_transform
        self.local_transform = local_transform

        # Per-instance volume cache. DataLoader workers are separate processes,
        # so each gets its own copy — no locking needed.
        self._cache_size = cache_size
        self._vol_cache: OrderedDict[str, np.ndarray] = OrderedDict()

        # (pid, mri_path, axis, slice_idx) — path stored per entry so multiple
        # phases of the same patient are handled correctly.
        self._index: list[tuple[int, Path, int, int]] = []

        for pid, mri_path in entries:
            try:
                img = nib.load(mri_path)
                shape = resampled_shape(img)
            except Exception:
                continue
            for axis in axes:
                for si in _sample_indices(shape[axis], n_per_axis):
                    self._index.append((pid, mri_path, axis, si))

    def __len__(self) -> int:
        return len(self._index)

    def _get_volume(self, mri_path: Path) -> np.ndarray:
        """Return the resampled (3D, float32) volume for mri_path, using the LRU cache."""
        key = str(mri_path)
        vol = self._vol_cache.get(key)
        if vol is not None:
            self._vol_cache.move_to_end(key)
            return vol

        img = nib.load(mri_path)
        vol = resample_to_spacing(img)
        # Handle 4D volumes (take first time point)
        if vol.ndim == 4:
            vol = vol[..., 0]
        vol = np.ascontiguousarray(vol, dtype=np.float32)

        if self._cache_size > 0:
            self._vol_cache[key] = vol
            if len(self._vol_cache) > self._cache_size:
                self._vol_cache.popitem(last=False)
        return vol

    def __getitem__(self, i: int) -> tuple[list[torch.Tensor], int]:
        pid, mri_path, axis, slice_idx = self._index[i]

        vol = self._get_volume(mri_path)

        slice_idx = min(slice_idx, vol.shape[axis] - 1)
        s = np.take(vol, slice_idx, axis=axis)

        # Normalize → (1, H, W) → resize to (3, img_size, img_size)
        t = torch.from_numpy(_normalize_slice(s)).unsqueeze(0)
        t = self.resize(t).repeat(3, 1, 1)

        views: list[torch.Tensor] = [
            self.global_transform(t),
            self.global_transform(t),
        ]
        for _ in range(self.n_local_crops):
            views.append(self.local_transform(t))

        return views, pid


def multicrop_collate(batch: list[tuple[list[torch.Tensor], int]]):
    """Collate a batch of (views_list, pid) items.

    Returns:
        views: list of n_crops tensors, each (B, C, H, W)
        pids: (B,) tensor of patient IDs
    """
    all_views = [item[0] for item in batch]
    pids = torch.tensor([item[1] for item in batch])
    n_crops = len(all_views[0])
    views = [torch.stack([all_views[b][c] for b in range(len(batch))]) for c in range(n_crops)]
    return views, pids
