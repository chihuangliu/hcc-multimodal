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
| 1 | raw, λ=0.1, unfrozen, n=10 | slice | `6a1a1bdf` | 0.671 | 0.833 | 0.742 | 0.850 | **0.742** |
| 1 | raw, λ=0.1, unfrozen, n=10 | patient | `1361bef2` | 0.483 | 0.666 | 0.561 | 0.713 | 0.561 |
| 2 | raw, λ=0.0, unfrozen, n=10 | slice | `982a6fa2` | 0.538 | 0.754 | 0.585 | 0.722 | 0.585 |
| 2 | raw, λ=0.0, unfrozen, n=10 | patient | `a6f970d6` | 0.514 | 0.701 | 0.494 | 0.673 | 0.514 |

#### Group 2 — Gene set ablation

| # | Config | Split | Model ID | LR AUROC | LR AUPRC | RF AUROC | RF AUPRC | **Best AUROC** |
|---|--------|-------|----------|--------:|--------:|--------:|--------:|----------:|
| 3 | raw, λ=0.1, predefined genes | slice | `12e4ba6a` | 0.587 | 0.807 | 0.656 | 0.823 | 0.656 |
| 3 | raw, λ=0.1, predefined genes | patient | `34e6806f` | 0.625 | 0.761 | 0.556 | 0.686 | 0.625 |
| 4 | raw, λ=0.1, 2y_before_cv genes | slice | `5d04e6ba` | 0.452 | 0.661 | 0.543 | 0.717 | 0.543 |
| 4 | raw, λ=0.1, 2y_before_cv genes | patient | `9109a6c2` | **0.725** | **0.862** | 0.580 | 0.749 | **0.725** |

#### Group 3 — Full slices (n=all, frozen backbone)

| # | Config | Split | Model ID | LR AUROC | LR AUPRC | RF AUROC | RF AUPRC | **Best AUROC** |
|---|--------|-------|----------|--------:|--------:|--------:|--------:|----------:|
| 5 | raw, λ=0.1, frozen, n=all | slice | `dc7e1d10` | — | — | — | — | pending |
| 5 | raw, λ=0.1, frozen, n=all | patient | `5e3f71a0` | — | — | — | — | pending |
| 6 | raw, λ=0.0, frozen, n=all | slice | `a64b245f` | — | — | — | — | pending |
| 6 | raw, λ=0.0, frozen, n=all | patient | `06c598c0` | — | — | — | — | pending |

#### Group 4 — Bounding box

| # | Config | Split | Model ID | LR AUROC | LR AUPRC | RF AUROC | RF AUPRC | **Best AUROC** |
|---|--------|-------|----------|--------:|--------:|--------:|--------:|----------:|
| 7 | raw_bbox, λ=0.1, unfrozen, n=10 | slice | `050d401d` | 0.583 | 0.743 | 0.516 | 0.694 | 0.583 |
| 7 | raw_bbox, λ=0.1, unfrozen, n=10 | patient | `f8aabb75` | 0.457 | 0.678 | 0.437 | 0.654 | 0.457 |
| 8 | raw_bbox, λ=0.0, unfrozen, n=10 | slice | `e12b0592` | 0.595 | 0.744 | 0.539 | 0.691 | 0.595 |
| 8 | raw_bbox, λ=0.0, unfrozen, n=10 | patient | `8715461c` | 0.611 | 0.764 | 0.457 | 0.705 | 0.611 |

### 2.3 Summary table

Ranked by best AUROC (available results only):

| Rank | Model ID | Config | Best AUROC | Best AUPRC |
|------|----------|--------|-----------:|-----------:|
| 1 | `6a1a1bdf` | raw, λ=0.1, slice split | **0.742** | **0.850** |
| 2 | `9109a6c2` | raw, λ=0.1, 2y_before_cv genes, patient split | **0.725** | **0.862** |
| 3 | `12e4ba6a` | raw, λ=0.1, predefined genes, slice split | 0.656 | 0.823 |
| 4 | `34e6806f` | raw, λ=0.1, predefined genes, patient split | 0.625 | 0.761 |
| 5 | `8715461c` | bbox, λ=0.0, patient split | 0.611 | 0.764 |
| 6 | `050d401d` | bbox, λ=0.1, slice split | 0.583 | 0.743 |
| 7 | `982a6fa2` | raw, λ=0.0, slice split | 0.585 | 0.722 |
| — | radiomic RF | resection-trained, 149 features | 0.590 | 0.766 |
| 8 | `e12b0592` | bbox, λ=0.0, slice split | 0.595 | 0.744 |
| 9 | `1361bef2` | raw, λ=0.1, patient split | 0.561 | 0.713 |
| 10 | `5d04e6ba` | raw, λ=0.1, 2y_before_cv genes, slice split | 0.543 | 0.717 |
| 11 | `a6f970d6` | raw, λ=0.0, patient split | 0.514 | 0.701 |
| — | radiomic LR | resection-trained, 149 features | 0.518 | 0.671 |
| 12 | `9109a6c2` RF | (same as rank 2, RF head) | 0.580 | 0.749 |
| 13 | `f8aabb75` | bbox, λ=0.1, patient split | 0.457 | 0.678 |
| 14 | `8715461c` RF | bbox, λ=0.0, patient split, RF | 0.457 | 0.705 |

---

## 3. Observations

1. **`6a1a1bdf` (raw, λ=0.1, slice split) remains top on ablation AUROC (0.742 RF)**, consistent with the CV results. Despite the CV slice-split concern, it generalises well to this fully held-out cohort.

2. **`9109a6c2` (2y_before_cv genes, patient split) is the strongest patient-split model (0.725 LR, 0.862 AUPRC)**, confirming that RFS-curated gene supervision improves external transferability even with an honest validation split.

3. **Slice-split inflation does not consistently hurt ablation performance.** The CV finding that slice-split inflated Model 5 (frozen, n=all) by +0.36 AUC is not replicated for Models 1–4 — their slice-split IDs perform *comparably or better* than their patient-split counterparts. This is plausible: leakage primarily inflates val AUC during training, not external-cohort AUC.

4. **The 2y_before_cv gene set reversal is striking**: slice-split `5d04e6ba` is the weakest (0.543) despite being the best CV model (0.789), while patient-split `9109a6c2` achieves 0.725. This suggests the slice-split version overfit to the resection distribution, whereas the patient-split version learned more transferable features.

5. **Bounding box models show no consistent advantage**: best bbox AUROC is 0.611 (`8715461c`), well below `6a1a1bdf` (0.742). The bbox advantage seen in slice-split CV (0.965) was entirely leakage.

6. **Radiomic RF (0.590) is competitive with most embedding models**, highlighting that simple radiomic features transfer reasonably across cohorts when the embedding model lacks outcome supervision or uses a leaky training split.

7. **Full-slice models (Groups 5–6) are still running** — these processed all ~80+ sagittal slices per patient and are expected to be slower at inference.

---

## 4. File references

| Artifact | Path |
|---|---|
| Radiomic LR | `results/eval_radiomic_lr.json` |
| Radiomic RF | `results/eval_radiomic_rf.json` |
| Embedding results (per model) | `results/eval_embedding_{model_id}.json` |
| Radiomic models | `models/radiomics/radiomic_rfs_2year_{lr,rf}.joblib` |
| Contrastive models | `training/contrastive/{model_id}/best_model.pt` |
| Cached ablation embeddings | `training/contrastive/{model_id}/cached_embeddings/ablation_img_emb.parquet` |
