
## Table of Contents

- [1. Task](#1-task)
- [2. Key findings](#2-key-findings)
- [3. Methodology](#3-methodology)
  - [3.1 Contrastive pre-training](#31-contrastive-pre-training)
  - [3.2 Embedding extraction](#32-embedding-extraction)
  - [3.3 Downstream CV pipeline](#33-downstream-cv-pipeline)
- [4. In-CV results](#4-in-cv-results)

# 1. Task

Replicate the 0518 in-CV experiments (λ=0.1 and λ=0) using **raw arterial-phase MRI** (`Resections_with_rna/MRI_liver_arterial.nii.gz`) instead of the preprocessed Radiomics/arterial images. The raw images are resampled to 1×1×3 mm (matching the preprocessed voxel spacing) but receive no further intensity pre-processing; per-slice normalisation (clip at 99th percentile, rescale to [0, 1]) is applied at load time as before.

All other hyperparameters are identical to the 0518 runs.

| Run | λ | MRI type | Model ID |
|-----|---|----------|----------|
| Raw λ=0.1 | 0.1 | `Resections_with_rna` | `6a1a1bdf` |
| Raw λ=0   | 0.0 | `Resections_with_rna` | `982a6fa2` |

| Task | Feature matrix | Dim |
|------|---------------|-----|
| radiomics | Arterial radiomics, SelectKBest F-score k=100 | 4132 → 100 |
| embeddings | Contrastive img+gene embeddings (128+128), SelectKBest k=100 | 256 → 100 |
| concat | Embeddings + arterial radiomics, SelectKBest k=100 | 4388 → 100 |
| ensemble | Embeddings model + radiomics model, probabilities averaged | — |

Each task run with SelectKBest placed **in-CV** only.

Results:
- Raw λ=0.1: `results/multimodal_prediction/<task>_6a1a1bdf_rfs_2year_in_cv_k100_raw_9ee3336/`
- Raw λ=0:   `results/multimodal_prediction/<task>_982a6fa2_rfs_2year_in_cv_k100_raw_9ee3336/`

---

# 2. Key findings

Best AUC across all tasks and models (in-CV):

| | Preprocessed (0518) | Raw (0525) |
|---|---|---|
| Radiomics baseline | RF: 0.569 ± 0.133 | RF: 0.569 ± 0.133 |
| λ=0.1 | Embeddings, RF: 0.706 ± 0.093 | Embeddings, RF: **0.798 ± 0.081** |
| λ=0   | Embeddings, RF: 0.752 ± 0.111 | Embeddings, LR: 0.706 ± 0.099 |

Raw MRI with λ=0.1 achieves the highest embeddings AUC (+0.09 over preprocessed). Unlike the preprocessed setting, λ=0.1 outperforms λ=0 on raw images. Concat and ensemble tasks do not benefit from raw embeddings compared to preprocessed.

---

# 3. Methodology

## 3.1 Contrastive pre-training

**Data**: `Resections_with_rna/MRI_liver_arterial.nii.gz`

**Raw image preprocessing** (applied at load time, per volume):
- Resample to 1×1×3 mm voxel spacing via linear interpolation (`scipy.ndimage.zoom`, order=1)
- No intensity windowing or normalisation at the volume level
- Per-slice: clip at 99th percentile, rescale to [0, 1] (same as preprocessed pipeline)

All encoder architecture and training settings match the 0518 runs:

| Parameter | `6a1a1bdf` (λ=0.1) | `982a6fa2` (λ=0) |
|-----------|-------------------|-----------------|
| Backbone | ViT-B/32 (unfrozen) | ViT-B/32 (unfrozen) |
| Embed dim | 128 per modality | 128 per modality |
| Gene hidden dim | 256 | 256 |
| Temperature τ | 0.07 | 0.07 |
| λ (reg weight) | 0.1 | 0.0 |
| Slices per patient | 10 (sagittal axis=0) | 10 (sagittal axis=0) |
| Image size | 224 × 224 | 224 × 224 |
| Epochs | 50 | 50 |
| Batch size | 32 | 32 |
| Optimiser | AdamW (lr=1e-4, wd=1e-4) | AdamW (lr=1e-4, wd=1e-4) |
| LR schedule | Cosine annealing (T_max=50) | Cosine annealing (T_max=50) |
| Val split | 10% stratified hold-out | 10% stratified hold-out |
| Checkpoint | Best validation loss | Best validation loss |
| Seed | 42 | 42 |
| Train loss (epoch 50) | 1.330 | 1.415 |
| Val loss (best) | 1.327 | 3.941 (early) |

The large train/val gap for λ=0 mirrors the 0518 preprocessed λ=0 run (val loss 4.00), consistent with NT-Xent overfitting without regularisation.

---

## 3.2 Embedding extraction

Identical to 0518: encoders frozen, 10 sagittal slices per patient mean-pooled → 128-dim image embedding, concatenated with 128-dim gene embedding → 256-dim joint embedding. 54 patients with paired MRI, RNA-seq, and clinical outcome.

---

## 3.3 Downstream CV pipeline

Identical to 0518 in-CV configuration: 3-fold stratified CV (`StratifiedKFold`, `random_state=42`), `SelectKBest(f_classif, k=100)` fitted on the training fold only, `StandardScaler` inside pipeline.

---

# 4. In-CV results

SelectKBest fitted on the training fold only. ROC-AUC mean ± std across 3 folds.

| Task | Model | Raw λ=0.1 AUC ± std | Raw λ=0 AUC ± std |
|------|-------|--------------------|--------------------|
| Radiomics | LR | 0.500 ± 0.000 | 0.500 ± 0.000 |
| Radiomics | RF | 0.569 ± 0.133 | 0.569 ± 0.133 |
| Embeddings | LR | 0.752 ± 0.062 | **0.706 ± 0.099** |
| Embeddings | RF | **0.798 ± 0.081** | 0.671 ± 0.065 |
| Concat | LR | 0.516 ± 0.092 | 0.529 ± 0.085 |
| Concat | RF | 0.489 ± 0.096 | 0.559 ± 0.144 |
| Ensemble | LR | 0.648 ± 0.096 | 0.578 ± 0.029 |
| Ensemble | RF | 0.696 ± 0.078 | 0.595 ± 0.063 |
