# Voxel Size Survey & Resampling
**Date:** 2026-06-22

---

## Native voxel spacing (before any processing)

| Dataset | File | x/y spacing (mm) | z spacing (mm) | z variants |
|---------|------|:-----------------:|:--------------:|:----------:|
| Resection (raw) | `Resections_with_rna/*/MRI_liver_arterial.nii.gz` | 0.62–1.95 | 2.0–4.0 | 7 |
| Soramic | `RFA_images/*/MRI_dyn_arterial.nii.gz` | 0.53–1.82 | 1.8–5.0 | 16 |
| Lausanne | `RFA_images/*/date_one/MRI_liver_arterial.nii.gz` | 0.55–1.60 | 2.0–4.5 | 11 |


---

## Processing pipeline

**Target spacing:** `(1.0, 1.0, 3.0)` mm — defined in `hcc_multimodal/contrastive/transform.py`.

### Step 1 — Resample to 1×1×3 mm (`resample_to_spacing`)

```
zoom_factor = native_spacing / target_spacing
output_shape = input_shape × zoom_factor
```

Applies `scipy.ndimage.zoom(order=1)` (linear interpolation). Physical FOV is preserved; only voxel count changes.

- Resection: pre-resampled and cached under `data/mri_resampled/` by `scripts/preresample_raw_mri.py`. Loaded directly at training time (`needs_resample=False`).
- Soramic / Lausanne: resampled on-the-fly at eval time (`resample=True` in `eval/data.py`).

### Step 2 — Slice and resize to 224×224

A 2D slice is taken along the sagittal axis, then resized to 224×224 with `torchvision.transforms.Resize` (antialias). This step discards FOV differences.

---

## Known issue: cache header bug

`scripts/preresample_raw_mri.py:33` saves cached files with `affine=np.eye(4)`, so the NIfTI header records spacing as 1×1×1 mm instead of 1×1×3 mm. The voxel data itself is correct (already resampled). No downstream effect because cached files bypass `resample_to_spacing`.
