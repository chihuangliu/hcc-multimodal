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
| `dc7e1d10` | raw, λ=0.1, frozen, n=all, slice | ⬜ pending |
| `5e3f71a0` | raw, λ=0.1, frozen, n=all, patient | ⬜ pending |
| `a64b245f` | raw, λ=0.0, frozen, n=all, slice | ⬜ pending |
| `06c598c0` | raw, λ=0.0, frozen, n=all, patient | ⬜ pending |
| `050d401d` | raw_bbox, λ=0.1, unfrozen, n=10, slice | ⬜ pending |
| `f8aabb75` | raw_bbox, λ=0.1, unfrozen, n=10, patient | ⬜ pending |
| `e12b0592` | raw_bbox, λ=0.0, unfrozen, n=10, slice | ⬜ pending |
| `8715461c` | raw_bbox, λ=0.0, unfrozen, n=10, patient | ⬜ pending |

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
| 5 | raw, λ=0.1, frozen, n=all | slice | `dc7e1d10` | 0.726 | 0.838 | 0.597 | 0.761 | **0.726** |
| 5 | raw, λ=0.1, frozen, n=all | patient | `5e3f71a0` | 0.603 | 0.766 | 0.671 | 0.834 | 0.671 |
| 6 | raw, λ=0.0, frozen, n=all | slice | `a64b245f` | 0.688 | 0.805 | 0.692 | 0.802 | 0.692 |
| 6 | raw, λ=0.0, frozen, n=all | patient | `06c598c0` | 0.708 | 0.826 | 0.667 | 0.842 | 0.708 |

#### Group 4 — Bounding box

| # | Config | Split | Model ID | LR AUROC | LR AUPRC | RF AUROC | RF AUPRC | **Best AUROC** |
|---|--------|-------|----------|--------:|--------:|--------:|--------:|----------:|
| 7 | raw_bbox, λ=0.1, unfrozen, n=10 | slice | `050d401d` | 0.583 | 0.743 | 0.516 | 0.694 | 0.583 |
| 7 | raw_bbox, λ=0.1, unfrozen, n=10 | patient | `f8aabb75` | 0.457 | 0.678 | 0.437 | 0.654 | 0.457 |
| 8 | raw_bbox, λ=0.0, unfrozen, n=10 | slice | `e12b0592` | 0.595 | 0.744 | 0.539 | 0.691 | 0.595 |
| 8 | raw_bbox, λ=0.0, unfrozen, n=10 | patient | `8715461c` | 0.611 | 0.764 | 0.457 | 0.705 | 0.611 |

### 2.3 Summary table

Ranked by best AUROC across LR and RF heads:

| Rank | Model ID | Config | Split | LR AUROC | RF AUROC | **Best AUROC** | Best AUPRC |
|------|----------|--------|-------|--------:|--------:|-----------:|-----------:|
| 1 | `6a1a1bdf` | raw, λ=0.1, n=10 | slice | 0.671 | **0.742** | **0.742** | 0.850 |
| 2 | `9109a6c2` | raw, λ=0.1, 2y_before_cv genes, n=10 | patient | **0.725** | 0.580 | **0.725** | **0.862** |
| 3 | `dc7e1d10` | raw, λ=0.1, frozen, n=all | slice | **0.726** | 0.597 | **0.726** | 0.838 |
| 4 | `06c598c0` | raw, λ=0.0, frozen, n=all | patient | **0.708** | 0.667 | **0.708** | 0.842 |
| 5 | `a64b245f` | raw, λ=0.0, frozen, n=all | slice | 0.688 | **0.692** | 0.692 | 0.805 |
| 6 | `5e3f71a0` | raw, λ=0.1, frozen, n=all | patient | 0.603 | **0.671** | 0.671 | 0.834 |
| 7 | `12e4ba6a` | raw, λ=0.1, predefined genes, n=10 | slice | 0.587 | **0.656** | 0.656 | 0.823 |
| — | radiomic RF | 149 art. features, resection-trained | — | — | 0.590 | 0.590 | 0.766 |
| 8 | `e12b0592` | bbox, λ=0.0, n=10 | slice | **0.595** | 0.539 | 0.595 | 0.744 |
| 9 | `982a6fa2` | raw, λ=0.0, n=10 | slice | 0.538 | **0.585** | 0.585 | 0.722 |
| 10 | `8715461c` | bbox, λ=0.0, n=10 | patient | **0.611** | 0.457 | 0.611 | 0.764 |
| 11 | `34e6806f` | raw, λ=0.1, predefined genes, n=10 | patient | **0.625** | 0.556 | 0.625 | 0.761 |
| — | radiomic LR | 149 art. features, resection-trained | — | 0.518 | — | 0.518 | 0.671 |
| 12 | `050d401d` | bbox, λ=0.1, n=10 | slice | **0.583** | 0.516 | 0.583 | 0.743 |
| 13 | `1361bef2` | raw, λ=0.1, n=10 | patient | 0.483 | **0.561** | 0.561 | 0.713 |
| 14 | `5d04e6ba` | raw, λ=0.1, 2y_before_cv genes, n=10 | slice | 0.452 | **0.543** | 0.543 | 0.717 |
| 15 | `a6f970d6` | raw, λ=0.0, n=10 | patient | **0.514** | 0.494 | 0.514 | 0.701 |
| 16 | `f8aabb75` | bbox, λ=0.1, n=10 | patient | **0.457** | 0.437 | 0.457 | 0.678 |

---

## 3. Observations

1. **Three models tie near the top (AUROC 0.72–0.74)**: `6a1a1bdf` (RF, 0.742), `9109a6c2` (LR, 0.725), and `dc7e1d10` (LR, 0.726). These come from different groups — λ=0.1 unfrozen n=10 slice-split, 2y_before_cv genes patient-split, and frozen n=all slice-split — suggesting multiple paths to good external transfer.

2. **Full-slice models (frozen backbone) transfer well.** `dc7e1d10` (0.726) and `06c598c0` (0.708) outperform most n=10 unfrozen models on ablation. Their CV slice-split AUC was 1.000 and 0.739 respectively — the `dc7e1d10` slice-split inflation did not hurt ablation AUROC, suggesting the frozen ViT-B/32 features themselves are transferable.

3. **The 2y_before_cv gene set reversal is striking**: slice-split `5d04e6ba` is the worst embedding model (0.543) despite being the top CV model (0.789), while its patient-split sibling `9109a6c2` achieves 0.725. The slice-split version likely overfit the resection gene-expression distribution; the patient-split version learned more transferable image features.

4. **Slice-split inflation does not consistently hurt ablation AUROC.** For most configs (Models 1, 2, 5, 6) the slice-split ID performs comparably or better than the patient-split counterpart. Leakage inflates internal val AUC during training; it does not necessarily cause overfitting to the resection domain.

5. **Bounding box models are consistently below raw-MRI equivalents**: best bbox is 0.611 (`8715461c`), vs 0.742 for the matched raw model. The slice-split bbox CV advantage (0.965) was entirely leakage.

6. **Radiomic RF (0.590) sits in the middle of the embedding model ranking**, ahead of 7 of the 16 embedding models. Simple arterial radiomics transfer non-trivially across cohorts when embedding models have poor supervision or leaky training.

7. **LR head tends to win for frozen/full-slice models; RF head wins for unfrozen/n=10 models.** `6a1a1bdf` RF beats LR by +0.07, while `dc7e1d10` LR beats RF by +0.13 — suggesting frozen backbone embeddings produce smoother, more linearly separable representations.

---

## 4. File references

| Artifact | Path |
|---|---|
| Radiomic LR | `results/eval/eval_radiomic_lr.json` |
| Radiomic RF | `results/eval/eval_radiomic_rf.json` |
| Embedding results (per model) | `results/eval/eval_embedding_{model_id}.json` |
| Radiomic models | `models/radiomics/radiomic_rfs_2year_{lr,rf}.joblib` |
| Contrastive models | `training/contrastive/{model_id}/best_model.pt` |
| Cached ablation embeddings | `training/contrastive/{model_id}/cached_embeddings/ablation_img_emb.parquet` |
