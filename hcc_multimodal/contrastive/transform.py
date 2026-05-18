"""Volume-level transforms for raw (unpreprocessed) MRI images."""

from __future__ import annotations

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

# Match the voxel spacing of the preprocessed Radiomics/arterial images.
TARGET_SPACING: tuple[float, float, float] = (1.0, 1.0, 3.0)


def resample_to_spacing(
    img: nib.Nifti1Image,
    target_spacing: tuple[float, float, float] = TARGET_SPACING,
) -> np.ndarray:
    """Resample a NIfTI volume to *target_spacing* (mm) using linear interpolation.

    Returns a float32 numpy array with the resampled voxel data.
    """
    native_spacing = np.array(img.header.get_zooms()[:3], dtype=float)
    target = np.array(target_spacing, dtype=float)
    zoom_factors = native_spacing / target

    data = np.asarray(img.dataobj, dtype=np.float32)
    if np.allclose(zoom_factors, 1.0, atol=1e-3):
        return data
    return zoom(data, zoom_factors, order=1).astype(np.float32)


def resampled_shape(
    img: nib.Nifti1Image,
    target_spacing: tuple[float, float, float] = TARGET_SPACING,
) -> tuple[int, ...]:
    """Return the spatial shape after resampling without loading voxel data."""
    native_spacing = np.array(img.header.get_zooms()[:3], dtype=float)
    target = np.array(target_spacing, dtype=float)
    zoom_factors = native_spacing / target
    return tuple(int(round(s * z)) for s, z in zip(img.shape[:3], zoom_factors))
