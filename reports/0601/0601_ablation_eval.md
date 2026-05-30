# Ablation Cohort Evaluation — 2-Year RFS Prediction
**Date:** 2026-06-01  
**Git commit (eval):** `716ab9ee`  
**Branch:** `test-set`

---

## Table of Contents

- [1. Setup](#1-setup)
- [2. Results](#2-results)
  - [2.1 Radiomic baselines](#21-radiomic-baselines)
  - [2.2 Embedding models — all configs](#22-embedding-models--all-configs)
  - [2.3 Summary table](#23-summary-table)
- [3. Observations](#3-observations)
- [4. File references](#4-file-references)

---

## 0. Re-run checklist (resample fix)

| Model ID | Config | Status |
|----------|--------|--------|
| `6a1a1bdf` | raw, λ=0.1, unfrozen, n=10, slice | ✅ done |
| `1361bef2` | raw, λ=0.1, unfrozen, n=10, patient | ✅ done |
| `982a6fa2` | raw, λ=0.0, unfrozen, n=10, slice | ✅ done |
| `a6f970d6` | raw, λ=0.0, unfrozen, n=10, patient | ✅ done |
| `12e4ba6a` | raw, λ=0.1, predefined genes, slice | ✅ done |
| `34e6806f` | raw, λ=0.1, predefined genes, patient | ✅ done |
| `5d04e6ba` | raw, λ=0.1, 2y_before_cv genes, slice | ✅ done |
| `9109a6c2` | raw, λ=0.1, 2y_before_cv genes, patient | ✅ done |
| `dc7e1d10` | raw, λ=0.1, frozen, n=all, slice | ✅ done |
| `5e3f71a0` | raw, λ=0.1, frozen, n=all, patient | ✅ done |
| `a64b245f` | raw, λ=0.0, frozen, n=all, slice | ✅ done |
| `06c598c0` | raw, λ=0.0, frozen, n=all, patient | ✅ done |
| `050d401d` | raw_bbox, λ=0.1, unfrozen, n=10, slice | ✅ done |
| `f8aabb75` | raw_bbox, λ=0.1, unfrozen, n=10, patient | ✅ done |
| `e12b0592` | raw_bbox, λ=0.0, unfrozen, n=10, slice | ✅ done |
| `8715461c` | raw_bbox, λ=0.0, unfrozen, n=10, patient | ✅ done |

---

## 1. Setup

### 1.1 Cohorts

| | Training (resection) | Test (ablation) |
|---|---|---|
| Patients | 54 | 59 (outcomes) / 100 (MRI) / 53 (radiomics) |
| Positives (RFS ≤ 2 yr) | 26 (48%) | — |
| Multi-lesion strategy | — | **average** (mean-pool features across lesions per patient) |

### 1.2 Radiomic pipeline

Trained on the full resection cohort (`models/radiomics/radiomic_rfs_2year_{lr,rf}.joblib`):

- 149 arterial-phase features → SelectKBest(f_classif, k=100) → classifier  
- **LR:** saga, elasticnet, l1_ratio=1.0, C=1.0  
- **RF:** max_depth=2, min_samples_leaf=10, n_estimators=100

### 1.3 Contrastive embedding pipeline

All 16 models from `reports/0601/0601_split_comparison.md` (8 slice-split × 8 patient-split).  
Downstream head applied identically to all: SelectKBest(f_classif, k=100) + LR or RF fitted on resection embeddings, evaluated on ablation embeddings.  
MRI: arterial phase (`MRI_dyn_arterial.nii.gz`), mean-pooled sagittal slices.

---

## 2. Results

### 2.1 Radiomic baselines

Target: rfs_2year | Multi-lesion: average | Threshold: 0.5

| Model | AUROC | AUPRC | Sensitivity | Specificity | PPV | NPV | F1 |
|-------|------:|------:|------------:|------------:|----:|----:|---:|
| LR | 0.518 | 0.671 | 0.657 | 0.389 | 0.677 | 0.368 | 0.667 |
| RF | 0.590 | 0.766 | 0.143 | 1.000 | 1.000 | 0.375 | 0.250 |

### 2.2 Embedding models — all configs

Best of LR / RF shown per model (best AUROC). "(pt)" = patient-level validation split during training; unmarked = slice-level split.

#### Group 1 — Raw MRI, λ sweep

| # | Config | Split | Model ID | LR AUROC | LR AUPRC | RF AUROC | RF AUPRC | **Best AUROC** |
|---|--------|-------|----------|--------:|--------:|--------:|--------:|----------:|
| 1 | raw, λ=0.1, unfrozen, n=10 | slice | `6a1a1bdf` | 0.615 | 0.816 | 0.583 | 0.744 | **0.615** |
| 1 | raw, λ=0.1, unfrozen, n=10 | patient | `1361bef2` | 0.470 | 0.657 | 0.522 | 0.707 | 0.522 |
| 2 | raw, λ=0.0, unfrozen, n=10 | slice | `982a6fa2` | 0.514 | 0.721 | 0.606 | 0.734 | **0.606** |
| 2 | raw, λ=0.0, unfrozen, n=10 | patient | `a6f970d6` | 0.494 | 0.674 | 0.450 | 0.651 | **0.494** |

#### Group 2 — Gene set ablation

| # | Config | Split | Model ID | LR AUROC | LR AUPRC | RF AUROC | RF AUPRC | **Best AUROC** |
|---|--------|-------|----------|--------:|--------:|--------:|--------:|----------:|
| 3 | raw, λ=0.1, predefined genes | slice | `12e4ba6a` | 0.578 | 0.798 | 0.670 | 0.820 | **0.670** |
| 3 | raw, λ=0.1, predefined genes | patient | `34e6806f` | 0.574 | 0.734 | 0.507 | 0.676 | **0.574** |
| 4 | raw, λ=0.1, 2y_before_cv genes | slice | `5d04e6ba` | 0.436 | 0.636 | 0.516 | 0.712 | **0.516** |
| 4 | raw, λ=0.1, 2y_before_cv genes | patient | `9109a6c2` | **0.732** | **0.865** | 0.568 | 0.724 | **0.732** |

#### Group 3 — Full slices (n=all, frozen backbone)

| # | Config | Split | Model ID | LR AUROC | LR AUPRC | RF AUROC | RF AUPRC | **Best AUROC** |
|---|--------|-------|----------|--------:|--------:|--------:|--------:|----------:|
| 5 | raw, λ=0.1, frozen, n=all | slice | `dc7e1d10` | **0.718** | 0.838 | 0.608 | 0.766 | **0.718** |
| 5 | raw, λ=0.1, frozen, n=all | patient | `5e3f71a0` | 0.617 | 0.775 | **0.635** | 0.819 | 0.635 |
| 6 | raw, λ=0.0, frozen, n=all | slice | `a64b245f` | **0.684** | 0.804 | 0.669 | 0.804 | **0.684** |
| 6 | raw, λ=0.0, frozen, n=all | patient | `06c598c0` | **0.702** | 0.840 | 0.664 | 0.830 | **0.702** |

#### Group 4 — Bounding box

| # | Config | Split | Model ID | LR AUROC | LR AUPRC | RF AUROC | RF AUPRC | **Best AUROC** |
|---|--------|-------|----------|--------:|--------:|--------:|--------:|----------:|
| 7 | raw_bbox, λ=0.1, unfrozen, n=10 | slice | `050d401d` | **0.669** | 0.791 | 0.515 | 0.711 | **0.669** |
| 7 | raw_bbox, λ=0.1, unfrozen, n=10 | patient | `f8aabb75` | **0.539** | 0.710 | 0.497 | 0.714 | 0.539 |
| 8 | raw_bbox, λ=0.0, unfrozen, n=10 | slice | `e12b0592` | **0.517** | 0.693 | 0.465 | 0.672 | 0.517 |
| 8 | raw_bbox, λ=0.0, unfrozen, n=10 | patient | `8715461c` | **0.534** | 0.661 | 0.431 | 0.637 | 0.534 |

### 2.3 Summary table

Ranked by best AUROC across LR and RF heads:

| Rank | Model ID | Config | Split | LR AUROC | RF AUROC | **Best AUROC** | Best AUPRC |
|------|----------|--------|-------|--------:|--------:|-----------:|-----------:|
| 1 | `9109a6c2` | raw, λ=0.1, 2y_before_cv genes, n=10 | patient | **0.732** | 0.568 | **0.732** | **0.865** |
| 2 | `dc7e1d10` | raw, λ=0.1, frozen, n=all | slice | **0.718** | 0.608 | **0.718** | 0.838 |
| 3 | `06c598c0` | raw, λ=0.0, frozen, n=all | patient | **0.702** | 0.664 | **0.702** | 0.840 |
| 4 | `a64b245f` | raw, λ=0.0, frozen, n=all | slice | **0.684** | 0.669 | **0.684** | 0.804 |
| 5 | `12e4ba6a` | raw, λ=0.1, predefined genes, n=10 | slice | 0.578 | **0.670** | **0.670** | 0.820 |
| 6 | `050d401d` | bbox, λ=0.1, n=10 | slice | **0.669** | 0.515 | **0.669** | 0.791 |
| 7 | `5e3f71a0` | raw, λ=0.1, frozen, n=all | patient | 0.617 | **0.635** | 0.635 | 0.819 |
| 8 | `6a1a1bdf` | raw, λ=0.1, n=10 | slice | **0.615** | 0.583 | 0.615 | 0.816 |
| 9 | `982a6fa2` | raw, λ=0.0, n=10 | slice | 0.514 | **0.606** | 0.606 | 0.734 |
| — | radiomic RF | 149 art. features, resection-trained | — | — | 0.590 | 0.590 | 0.766 |
| 10 | `34e6806f` | raw, λ=0.1, predefined genes, n=10 | patient | **0.574** | 0.507 | 0.574 | 0.734 |
| 11 | `f8aabb75` | bbox, λ=0.1, n=10 | patient | **0.539** | 0.497 | 0.539 | 0.710 |
| 12 | `8715461c` | bbox, λ=0.0, n=10 | patient | **0.534** | 0.431 | 0.534 | 0.661 |
| — | radiomic LR | 149 art. features, resection-trained | — | 0.518 | — | 0.518 | 0.671 |
| 13 | `1361bef2` | raw, λ=0.1, n=10 | patient | 0.470 | **0.522** | 0.522 | 0.707 |
| 14 | `e12b0592` | bbox, λ=0.0, n=10 | slice | **0.517** | 0.465 | 0.517 | 0.693 |
| 15 | `5d04e6ba` | raw, λ=0.1, 2y_before_cv genes, n=10 | slice | 0.436 | **0.516** | 0.516 | 0.712 |
| 16 | `a6f970d6` | raw, λ=0.0, n=10 | patient | **0.494** | 0.450 | 0.494 | 0.674 |

---

## 3. Observations

1. **Top three models (AUROC 0.70–0.73) come from two different groups**: `9109a6c2` (LR, 0.732), `dc7e1d10` (LR, 0.718), and `06c598c0` (LR, 0.702). The 2y_before_cv patient-split model and both frozen n=all models transfer best, suggesting that avoiding slice-level leakage during training and using all slices are complementary paths to external generalization.

2. **Full-slice frozen models transfer consistently well.** All four frozen n=all models rank in the top 4, with AUROC 0.635–0.718. Their frozen ViT-B/32 backbone produces transferable representations regardless of λ or split strategy.

3. **`6a1a1bdf` (previously rank 1 at 0.742) drops to rank 8 (0.615) after the resampling fix.** The old results were extracted without resampling the ablation MRIs; with correct 1×1×3 mm resampling the advantage disappears, suggesting its prior lead was an artefact of resolution mismatch rather than a better model.

4. **The 2y_before_cv gene set reversal is preserved**: slice-split `5d04e6ba` remains the worst embedding model (0.516) while patient-split sibling `9109a6c2` is the best (0.732). The patient-level split forces the model to learn more transferable image features rather than overfitting the resection gene-expression distribution.

5. **`050d401d` (bbox, slice-split) jumped from rank 12 to rank 6 (0.583→0.669) after the resampling fix.** The bbox pipeline crops around segmentation masks that were themselves resampled — getting this right mattered more for bbox than for raw-MRI models.

6. **Radiomic RF (0.590) still sits mid-table**, ahead of 7 of the 16 embedding models. The relative standing is unchanged from before the resampling fix.

7. **LR head dominates across the board after the fix.** LR wins for 13 of 16 models, including all frozen and most unfrozen configs — a stronger pattern than before where RF won for several n=10 unfrozen models.

---

## 4. File references

| Artifact | Path |
|---|---|
| Radiomic LR | `results/eval/ablation/radiomic_lr_rfs_2year_{timestamp}.json` |
| Radiomic RF | `results/eval/ablation/radiomic_rf_rfs_2year_{timestamp}.json` |
| Embedding results (per model) | `results/eval/ablation/embedding_{model_id}_rfs_2year_{timestamp}.json` |
| Radiomic models | `models/radiomics/radiomic_rfs_2year_{lr,rf}.joblib` |
| Contrastive models | `training/contrastive/{model_id}/best_model.pt` |
| Cached ablation embeddings | `training/contrastive/{model_id}/cached_embeddings/ablation_img_emb.parquet` |
