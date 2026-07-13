# Ablation Cohort Evaluation — 2-Year RFS Prediction (v3)
**Date:** 2026-07-13  
**Covers:** Soramic (ablation) cohort · Lausanne cohort  
**Based on:** `reports/0608/0608_ablation_eval_v2.md`  
**Change vs v2:** §4 "Rank by CV AUC" recomputed on the **image-only** `resection_img_emb.parquet` embedding — the *same* extraction as the §2–3 transfer columns — with fixed LR/RF, **no feature selection**, plain 3-fold CV; the v2 nested grid-search / gene-leaked CV table is removed.

---

## Table of Contents

- [1. Setup](#1-setup)
- [2. Results — Soramic](#2-results--soramic)
  - [2.1 Radiomic baselines](#21-radiomic-baselines)
  - [2.2 Embedding models — all configs](#22-embedding-models--all-configs)
  - [2.3 Summary table](#23-summary-table)
- [3X. Results — Lausanne](#3-results--lausanne)
  - [3.1 Radiomic baselines](#31-radiomic-baselines)
  - [3.2 Embedding models — all configs](#32-embedding-models--all-configs)
  - [3.3 Summary table](#33-summary-table)
- [4. Rank by CV AUC](#4-rank-by-cv-auc)
- [5. Observations](#5-observations)
- [6. Ensemble results](#6-ensemble-results)
  - [6.1 Soramic — ensemble summary](#61-soramic--ensemble-summary)
  - [6.2 Lausanne — ensemble summary](#62-lausanne--ensemble-summary)
- [7. File references](#7-file-references)

---

## 1. Setup

### 1.1 Cohorts

| | Training (resection) | Test (Soramic) | Test (Lausanne) |
|---|---|---|---|
| Patients | 54 | 59 with 2 yr RFS outcome, 53 with radiomics features | 66 with 2 yr RFS outcome, 61 with radiomics features |
| Positives (RFS ≤ 2 yr) | 26 (48%) | 40 (68%) | 49 (74%) |

### 1.2 Radiomic pipeline

Pre-trained on the full resection cohort (`models/radiomics/radiomic_rfs_2year_{lr,rf}.joblib`):

- 149 arterial-phase features → SelectKBest(f_classif, k=100) → classifier  
- **LR:** saga, elasticnet, l1_ratio=1.0, C=1.0  
- **RF:** max_depth=2, min_samples_leaf=10, n_estimators=100

### 1.3 Contrastive embedding pipeline

Same 16 models evaluated on both cohorts (Groups 1–4). Downstream head: SelectKBest(f_classif, k=100) + LR or RF fitted on resection embeddings, evaluated on test cohort embeddings. MRI: arterial phase, mean-pooled sagittal slices. Embeddings resampled to 1×1×3 mm before extraction.

#### 1.3.1 Contrastive training configurations

All 16 models share the same ViT-B/32 backbone and triplet contrastive objective. The four axes varied across the ablation are described below.

| Parameter | Values | Notes |
|---|---|---|
| **λ** | 0.0, 0.1 | Weight of the contrastive loss relative to the supervised classification loss. λ=0.0 trains without contrastive signal. |
| **Frozen backbone** | frozen, unfrozen | Whether the ViT-B/32 weights are kept fixed (only the projection head trains) or fine-tuned end-to-end. |
| **n slices per patient** | 10, all | Number of axial slices sampled per patient during training. "all" uses every sagittal slice available. |
| **Validation split** | slice, patient | Unit of train/val split during training. Slice-level split may allow the same patient's slices to appear in both train and val; patient-level split holds out full patients. |

The 16 model configurations:

| Group | Model ID | Input | Gene set | λ | Frozen | n slices | Val split |
|-------|----------|-------|----------|---|--------|----------|-----------|
| 1 | `6a1a1bdf` | raw | 40 genes | 0.1 | no | 10 | slice |
| 1 | `1361bef2` | raw | 40 genes | 0.1 | no | 10 | patient |
| 1 | `982a6fa2` | raw | 40 genes | 0.0 | no | 10 | slice |
| 1 | `a6f970d6` | raw | 40 genes | 0.0 | no | 10 | patient |
| 2 | `12e4ba6a` | raw | predefined | 0.1 | no | 10 | slice |
| 2 | `34e6806f` | raw | predefined | 0.1 | no | 10 | patient |
| 2 | `5d04e6ba` | raw | 2y\_before\_cv | 0.1 | no | 10 | slice |
| 2 | `9109a6c2` | raw | 2y\_before\_cv | 0.1 | no | 10 | patient |
| 3 | `dc7e1d10` | raw | 40 genes | 0.1 | yes | all | slice |
| 3 | `5e3f71a0` | raw | 40 genes | 0.1 | yes | all | patient |
| 3 | `a64b245f` | raw | 40 genes | 0.0 | yes | all | slice |
| 3 | `06c598c0` | raw | 40 genes | 0.0 | yes | all | patient |
| 4 | `050d401d` | bbox | 40 genes | 0.1 | no | 10 | slice |
| 4 | `f8aabb75` | bbox | 40 genes | 0.1 | no | 10 | patient |
| 4 | `e12b0592` | bbox | 40 genes | 0.0 | no | 10 | slice |
| 4 | `8715461c` | bbox | 40 genes | 0.0 | no | 10 | patient |
| 4 | `92b9afed` | bbox | 40 genes | 0.1 | yes | all | slice |

Group 1: baseline raw-MRI configs; Group 2: gene-set ablation (same backbone/training as Group 1); Group 3: frozen backbone with all slices; Group 4: bounding-box crop input.

### 1.4 Key differences between cohorts

| | Soramic (ablation) | Lausanne |
|---|---|---|
| MRI filename | `{pid}/MRI_dyn_arterial.nii.gz` | `{pid:04d}/date_one/MRI_liver_arterial.nii.gz` |
| Mask prefix | `{pid}` | `{pid:03d}` |
| N patients (outcome) | 59 | 66 |
| Positive rate | 68% | 74% |

---

## 2. Results — Soramic

### 2.1 Radiomic baselines

Target: rfs_2year | Multi-lesion: average | Threshold: 0.5

| Model | AUROC | AUPRC | Sensitivity | Specificity | PPV | NPV | F1 |
|-------|------:|------:|------------:|------------:|----:|----:|---:|
| LR | 0.518 | 0.671 | 0.657 | 0.389 | 0.677 | 0.368 | 0.667 |
| RF | **0.590** | 0.766 | 0.143 | 1.000 | 1.000 | 0.375 | 0.250 |

### 2.2 Embedding models — all configs

Best of LR / RF shown per model (best AUROC).

#### Group 1 — 40 genes + 10 slices per patient

| Config | Model ID | LR AUROC | RF AUROC | Best AUROC |
|--------|----------|--------:|--------:|----------:|
| λ=0.1, unfrozen, n=10, slice | `6a1a1bdf` | 0.615 | 0.583 | 0.615 |
| λ=0.1, unfrozen, n=10, patient | `1361bef2` | 0.470 | 0.522 | 0.522 |
| λ=0.0, unfrozen, n=10, slice | `982a6fa2` | 0.514 | 0.606 | 0.606 |
| λ=0.0, unfrozen, n=10, patient | `a6f970d6` | 0.494 | 0.450 | 0.494 |

#### Group 2 — Gene set ablation

| Config | Model ID | LR AUROC | RF AUROC | Best AUROC |
|--------|----------|--------:|--------:|----------:|
| λ=0.1, predefined genes, slice | `12e4ba6a` | 0.578 | 0.670 | 0.670 |
| λ=0.1, predefined genes, patient | `34e6806f` | 0.574 | 0.507 | 0.574 |
| λ=0.1, 2y_before_cv genes, slice | `5d04e6ba` | 0.436 | 0.516 | 0.516 |
| λ=0.1, 2y_before_cv genes, patient | `9109a6c2` | 0.732 | 0.568 | 0.732 |

#### Group 3 — Full slices (n=all, frozen backbone)

| Config | Model ID | LR AUROC | RF AUROC | Best AUROC |
|--------|----------|--------:|--------:|----------:|
| λ=0.1, frozen, n=all, slice | `dc7e1d10` | 0.718 | 0.608 | 0.718 |
| λ=0.1, frozen, n=all, patient | `5e3f71a0` | 0.617 | 0.635 | 0.635 |
| λ=0.0, frozen, n=all, slice | `a64b245f` | 0.684 | 0.669 | 0.684 |
| λ=0.0, frozen, n=all, patient | `06c598c0` | 0.702 | 0.664 | 0.702 |

#### Group 4 — Bounding box

| Config | Model ID | LR AUROC | RF AUROC | Best AUROC |
|--------|----------|--------:|--------:|----------:|
| λ=0.1, unfrozen, n=10, slice | `050d401d` | 0.669 | 0.515 | 0.669 |
| λ=0.1, unfrozen, n=10, patient | `f8aabb75` | 0.539 | 0.497 | 0.539 |
| λ=0.0, unfrozen, n=10, slice | `e12b0592` | 0.517 | 0.465 | 0.517 |
| λ=0.0, unfrozen, n=10, patient | `8715461c` | 0.534 | 0.431 | 0.534 |
| λ=0.1, frozen, n=all, slice | `92b9afed` | 0.571 | 0.577 | 0.577 |

### 2.3 Summary table

Best head (LR or RF, whichever has higher AUROC) per model, ranked by AUROC. Multi-lesion: average. Threshold: 0.5. "—" = undefined NPV (all samples predicted positive).

| Rank | Model ID | Head | Config | AUROC | AUPRC | Sensitivity | Specificity | PPV | NPV | F1 |
|------|----------|------|--------|------:|------:|------------:|------------:|----:|----:|---:|
| 1 | `9109a6c2` | LR | raw, λ=0.1, 2y_before_cv genes, n=10, patient | 0.732 | 0.865 | 0.846 | 0.389 | 0.750 | 0.538 | 0.795 |
| 2 | `dc7e1d10` | LR | raw, λ=0.1, frozen, n=all, slice | 0.718 | 0.838 | 0.846 | 0.389 | 0.750 | 0.538 | 0.795 |
| 3 | `06c598c0` | LR | raw, λ=0.0, frozen, n=all, patient | 0.702 | 0.840 | 0.897 | 0.222 | 0.714 | 0.500 | 0.795 |
| 4 | `a64b245f` | LR | raw, λ=0.0, frozen, n=all, slice | 0.684 | 0.804 | 0.974 | 0.278 | 0.745 | 0.833 | 0.844 |
| 5 | `12e4ba6a` | RF | raw, λ=0.1, predefined genes, n=10, slice | 0.670 | 0.820 | 0.872 | 0.278 | 0.723 | 0.500 | 0.791 |
| 6 | `050d401d` | LR | bbox, λ=0.1, unfrozen, n=10, slice | 0.669 | 0.791 | 1.000 | 0.000 | 0.667 | — | 0.800 |
| 7 | `5e3f71a0` | RF | raw, λ=0.1, frozen, n=all, patient | 0.635 | 0.819 | 1.000 | 0.000 | 0.684 | — | 0.812 |
| 8 | `6a1a1bdf` | LR | raw, λ=0.1, unfrozen, n=10, slice | 0.615 | 0.816 | 0.385 | 0.778 | 0.789 | 0.368 | 0.517 |
| 9 | `982a6fa2` | RF | raw, λ=0.0, unfrozen, n=10, slice | 0.606 | 0.734 | 1.000 | 0.000 | 0.684 | — | 0.812 |
| — | radiomic RF | RF | 149 art. features, resection-trained | 0.590 | 0.766 | 0.143 | 1.000 | 1.000 | 0.375 | 0.250 |
| 10 | `92b9afed` | RF | bbox, λ=0.1, frozen, n=all, slice | 0.577 | 0.719 | 0.972 | 0.000 | 0.660 | 0.000 | 0.787 |
| 11 | `34e6806f` | LR | raw, λ=0.1, predefined genes, n=10, patient | 0.574 | 0.734 | 0.667 | 0.500 | 0.743 | 0.409 | 0.703 |
| 12 | `f8aabb75` | LR | bbox, λ=0.1, unfrozen, n=10, patient | 0.539 | 0.710 | 0.944 | 0.056 | 0.667 | 0.333 | 0.782 |
| 13 | `8715461c` | LR | bbox, λ=0.0, unfrozen, n=10, patient | 0.534 | 0.661 | 0.028 | 0.889 | 0.333 | 0.314 | 0.051 |
| 14 | `1361bef2` | RF | raw, λ=0.1, unfrozen, n=10, patient | 0.522 | 0.707 | 1.000 | 0.000 | 0.684 | — | 0.812 |
| — | radiomic LR | LR | 149 art. features, resection-trained | 0.518 | 0.671 | 0.657 | 0.389 | 0.676 | 0.368 | 0.667 |
| 15 | `e12b0592` | LR | bbox, λ=0.0, unfrozen, n=10, slice | 0.517 | 0.693 | 0.472 | 0.556 | 0.680 | 0.345 | 0.557 |
| 16 | `5d04e6ba` | RF | raw, λ=0.1, 2y_before_cv genes, n=10, slice | 0.516 | 0.712 | 1.000 | 0.000 | 0.684 | — | 0.812 |
| 17 | `a6f970d6` | LR | raw, λ=0.0, unfrozen, n=10, patient | 0.494 | 0.674 | 0.949 | 0.056 | 0.685 | 0.333 | 0.796 |

---

## 3. Results — Lausanne

### 3.1 Radiomic baselines

Target: rfs_2year | Multi-lesion: average | Threshold: 0.5

| Model | AUROC | AUPRC | Sensitivity | Specificity | PPV | NPV | F1 |
|-------|------:|------:|------------:|------------:|----:|----:|---:|
| LR | 0.531 | 0.748 | 0.422 | 0.625 | 0.760 | 0.278 | 0.543 |
| RF | 0.506 | 0.747 | 0.067 | 0.875 | 0.600 | 0.250 | 0.120 |

### 3.2 Embedding models — all configs

Best of LR / RF shown per model (best AUROC).

#### Group 1 — 40 genes + 10 slices per patient

| Config | Model ID | LR AUROC | RF AUROC | Best AUROC |
|--------|----------|--------:|--------:|----------:|
| λ=0.1, unfrozen, n=10, slice | `6a1a1bdf` | 0.497 | 0.474 | 0.497 |
| λ=0.1, unfrozen, n=10, patient | `1361bef2` | 0.664 | **0.771** | **0.771** |
| λ=0.0, unfrozen, n=10, slice | `982a6fa2` | 0.600 | 0.560 | 0.600 |
| λ=0.0, unfrozen, n=10, patient | `a6f970d6` | 0.618 | 0.463 | 0.618 |

#### Group 2 — Gene set ablation

| Config | Model ID | LR AUROC | RF AUROC | Best AUROC |
|--------|----------|--------:|--------:|----------:|
| λ=0.1, predefined genes, slice | `12e4ba6a` | 0.423 | 0.477 | 0.477 |
| λ=0.1, predefined genes, patient | `34e6806f` | 0.420 | 0.358 | 0.420 |
| λ=0.1, 2y_before_cv genes, slice | `5d04e6ba` | 0.655 | 0.610 | 0.655 |
| λ=0.1, 2y_before_cv genes, patient | `9109a6c2` | 0.563 | 0.414 | 0.563 |

#### Group 3 — Full slices (n=all, frozen backbone)

| Config | Model ID | LR AUROC | RF AUROC | Best AUROC |
|--------|----------|--------:|--------:|----------:|
| λ=0.1, frozen, n=all, slice | `dc7e1d10` | 0.419 | 0.453 | 0.453 |
| λ=0.1, frozen, n=all, patient | `5e3f71a0` | 0.534 | 0.467 | 0.534 |
| λ=0.0, frozen, n=all, slice | `a64b245f` | 0.448 | 0.556 | 0.556 |
| λ=0.0, frozen, n=all, patient | `06c598c0` | 0.515 | 0.450 | 0.515 |

#### Group 4 — Bounding box

| Config | Model ID | LR AUROC | RF AUROC | Best AUROC |
|--------|----------|--------:|--------:|----------:|
| λ=0.1, unfrozen, n=10, slice | `050d401d` | 0.502 | 0.544 | 0.544 |
| λ=0.1, unfrozen, n=10, patient | `f8aabb75` | 0.446 | 0.515 | 0.515 |
| λ=0.0, unfrozen, n=10, slice | `e12b0592` | 0.489 | 0.595 | 0.595 |
| λ=0.0, unfrozen, n=10, patient | `8715461c` | 0.490 | 0.494 | 0.494 |
| λ=0.1, frozen, n=all, slice | `92b9afed` | 0.607 | 0.614 | **0.614** |

### 3.3 Summary table

Best head (LR or RF, whichever has higher AUROC) per model, ranked by AUROC. Multi-lesion: average. Threshold: 0.5. "—" = undefined NPV (all samples predicted positive).

| Rank | Model ID | Head | Config | AUROC | AUPRC | Sensitivity | Specificity | PPV | NPV | F1 |
|------|----------|------|--------|------:|------:|------------:|------------:|----:|----:|---:|
| 1 | `1361bef2` | RF | raw, λ=0.1, unfrozen, n=10, patient | **0.771** | 0.867 | 1.000 | 0.000 | 0.742 | — | 0.852 |
| 2 | `92b9afed` | RF | bbox, λ=0.1, frozen, n=all, slice | 0.614 | 0.810 | 0.979 | 0.059 | 0.746 | 0.500 | 0.847 |
| 3 | `5d04e6ba` | LR | raw, λ=0.1, 2y_before_cv genes, n=10, slice | 0.655 | 0.850 | 1.000 | 0.000 | 0.742 | — | 0.852 |
| 4 | `a6f970d6` | LR | raw, λ=0.0, unfrozen, n=10, patient | 0.618 | 0.827 | 0.980 | 0.059 | 0.750 | 0.500 | 0.850 |
| 5 | `982a6fa2` | LR | raw, λ=0.0, unfrozen, n=10, slice | 0.600 | 0.844 | 0.980 | 0.059 | 0.750 | 0.500 | 0.850 |
| 6 | `e12b0592` | RF | bbox, λ=0.0, unfrozen, n=10, slice | 0.595 | 0.790 | 0.438 | 0.706 | 0.808 | 0.308 | 0.568 |
| 7 | `9109a6c2` | LR | raw, λ=0.1, 2y_before_cv genes, n=10, patient | 0.563 | 0.806 | 0.694 | 0.353 | 0.756 | 0.286 | 0.723 |
| 8 | `a64b245f` | RF | raw, λ=0.0, frozen, n=all, slice | 0.556 | 0.803 | 0.959 | 0.059 | 0.746 | 0.333 | 0.839 |
| 9 | `050d401d` | RF | bbox, λ=0.1, unfrozen, n=10, slice | 0.544 | 0.755 | 0.333 | 0.765 | 0.800 | 0.289 | 0.471 |
| — | radiomic LR | LR | 149 art. features, resection-trained | 0.531 | 0.748 | 0.422 | 0.625 | 0.760 | 0.278 | 0.543 |
| 10 | `5e3f71a0` | LR | raw, λ=0.1, frozen, n=all, patient | 0.534 | 0.775 | 0.939 | 0.118 | 0.754 | 0.400 | 0.836 |
| 11 | `f8aabb75` | RF | bbox, λ=0.1, unfrozen, n=10, patient | 0.515 | 0.769 | 1.000 | 0.000 | 0.738 | — | 0.850 |
| 12 | `06c598c0` | LR | raw, λ=0.0, frozen, n=all, patient | 0.515 | 0.779 | 0.755 | 0.294 | 0.755 | 0.294 | 0.755 |
| — | radiomic RF | RF | 149 art. features, resection-trained | 0.506 | 0.747 | 0.067 | 0.875 | 0.600 | 0.250 | 0.120 |
| 13 | `6a1a1bdf` | LR | raw, λ=0.1, unfrozen, n=10, slice | 0.497 | 0.753 | 0.204 | 0.765 | 0.714 | 0.250 | 0.317 |
| 14 | `8715461c` | RF | bbox, λ=0.0, unfrozen, n=10, patient | 0.494 | 0.766 | 0.646 | 0.353 | 0.738 | 0.261 | 0.689 |
| 15 | `dc7e1d10` | RF | raw, λ=0.1, frozen, n=all, slice | 0.453 | 0.727 | 0.776 | 0.118 | 0.717 | 0.154 | 0.745 |
| 16 | `12e4ba6a` | RF | raw, λ=0.1, predefined genes, n=10, slice | 0.477 | 0.755 | 0.796 | 0.176 | 0.736 | 0.231 | 0.765 |
| 17 | `34e6806f` | LR | raw, λ=0.1, predefined genes, n=10, patient | 0.420 | 0.702 | 0.531 | 0.412 | 0.722 | 0.233 | 0.612 |

---

## 4. Rank by CV AUC

In-CV AUC uses **fixed LR/RF, no hyperparameter search, no feature selection**, all on the **same
image extraction as the transfer columns** — the survival `resection_img_emb.parquet` cache (128-dim
image-only, patient-level mean-pooled). LR/RF are taken as-is from `hcc_multimodal/baselines/config.py`
(`MODELS`; LR = saga, elasticnet, l1_ratio=1.0, C=1.0; RF = 100 trees, default depth) and fit on
**all 128 image dims** through `SimpleImputer(median) → StandardScaler → classifier` in a plain
**3-fold** stratified CV on the 54-patient (26 positive) resection cohort. "Best head" = higher mean
fold AUC. Soramic/Lausanne columns are the best-head transfer values from §2–3 (same cache); Δ vs.
best radiomic baseline (Soramic RF=0.590; Lausanne LR=0.531).

> **Note — one extraction throughout.** Earlier drafts computed this CV on the `multimodal_prediction`
> `emb_*.parquet` cache, a *different* extraction from the §2–3 transfer cache (per-patient vectors
> only ~0.77 cosine apart), which inflated the most-separable slice-split models to CV≈1.000. CV,
> transfer, and the companion §6 grid now all read `resection_img_emb.parquet`, so the columns are
> directly comparable and the spurious 1.000s are gone (`dc7e1d10` → 0.695).

> **Convergence note (2026-07-20).** The LR head's `max_iter` was bumped 1000 → 5000 so the saga
> solver converges on the 128-dim embedding. This lowers a few LR CV values slightly (`dc7e1d10`
> 0.699 → **0.695**, `982a6fa2` 0.682 → 0.677) and flips three best-heads at convergence
> (`92b9afed` RF→LR, `5e3f71a0` LR→RF, `f8aabb75` LR→RF). Soramic/Lausanne transfer columns (§2–3,
> a separate SelectKBest pipeline) are unchanged. `dc7e1d10` 0.695 now matches the
> `LASSO`/`All features` grid anchor in `reports/0720/0720_embedding_grid_eval_v3.md`.

| CV Rank | Model ID | Config | Best head | CV AUC ± std | Soramic AUROC | Lausanne AUROC |
|--------:|----------|--------|-----------|-------------:|--------------:|---------------:|
| 1 | `a6f970d6` | raw, λ=0.0, unfrozen, n=10, patient | LR | **0.714 ± 0.133** | 0.494 (−0.096) | 0.618 (+0.087) |
| 2 | `dc7e1d10` | raw, λ=0.1, frozen, n=all, slice | LR | 0.695 ± 0.117 | 0.718 (+0.128) | 0.453 (−0.078) |
| 3 | `982a6fa2` | raw, λ=0.0, unfrozen, n=10, slice | LR | 0.677 ± 0.038 | 0.606 (+0.016) | 0.600 (+0.069) |
| 4 | `a64b245f` | raw, λ=0.0, frozen, n=all, slice | LR | 0.665 ± 0.092 | 0.684 (+0.094) | 0.556 (+0.025) |
| 5 | `92b9afed` | bbox, λ=0.1, frozen, n=all, slice | LR | 0.662 ± 0.063 | 0.577 (−0.013) | 0.614 (+0.083) |
| 6 | `1361bef2` | raw, λ=0.1, unfrozen, n=10, patient | LR | 0.645 ± 0.014 | 0.522 (−0.068) | **0.771 (+0.240)** |
| 7 | `5e3f71a0` | raw, λ=0.1, frozen, n=all, patient | RF | 0.616 ± 0.010 | 0.635 (+0.045) | 0.534 (+0.003) |
| 8 | `06c598c0` | raw, λ=0.0, frozen, n=all, patient | LR | 0.603 ± 0.160 | 0.702 (+0.112) | 0.515 (−0.016) |
| 9 | `12e4ba6a` | raw, λ=0.1, predefined genes, slice | LR | 0.595 ± 0.121 | 0.670 (+0.080) | 0.477 (−0.054) |
| 10 | `5d04e6ba` | raw, λ=0.1, 2y_before_cv genes, slice | LR | 0.591 ± 0.073 | 0.516 (−0.074) | 0.655 (+0.124) |
| 11 | `050d401d` | bbox, λ=0.1, unfrozen, n=10, slice | LR | 0.579 ± 0.161 | 0.669 (+0.079) | 0.544 (+0.013) |
| 12 | `8715461c` | bbox, λ=0.0, unfrozen, n=10, patient | LR | 0.579 ± 0.087 | 0.534 (−0.056) | 0.494 (−0.037) |
| 13 | `e12b0592` | bbox, λ=0.0, unfrozen, n=10, slice | LR | 0.562 ± 0.074 | 0.517 (−0.073) | 0.595 (+0.064) |
| 14 | `6a1a1bdf` | raw, λ=0.1, unfrozen, n=10, slice | RF | 0.554 ± 0.031 | 0.615 (+0.025) | 0.497 (−0.034) |
| 15 | `9109a6c2` | raw, λ=0.1, 2y_before_cv genes, patient | LR | 0.545 ± 0.028 | **0.732 (+0.142)** | 0.563 (+0.032) |
| 16 | `34e6806f` | raw, λ=0.1, predefined genes, patient | LR | 0.512 ± 0.097 | 0.574 (−0.016) | 0.420 (−0.111) |
| 17 | `f8aabb75` | bbox, λ=0.1, unfrozen, n=10, patient | RF | 0.502 ± 0.076 | 0.539 (−0.051) | 0.515 (−0.016) |

**Resection CV does not point to a transferable embedding.** The CV-top `a6f970d6` (0.714) is
**chance on Soramic (0.494)** — below the radiomic baseline. Across the 17 models CV is uncorrelated
with Soramic transfer (Spearman **0.06**), weakly positive with Lausanne (0.36), and the two cohorts
anti-correlate (Soramic↔Lausanne **−0.46**). The only high-CV model that also transfers on Soramic is
`dc7e1d10` (CV rank 2, 0.718), but it is near-worst on Lausanne (0.453); `9109a6c2` tops Soramic
(0.732) from CV rank 15. A per-head FS × classifier grid on `dc7e1d10` is in the companion
`reports/0713/0713_embedding_grid_eval_v2.md` §6.

---

## 5. Observations

### 5.1 Soramic

1. **Top three models (AUROC 0.70–0.73) come from two different groups**: `9109a6c2` (LR, 0.732), `dc7e1d10` (LR, 0.718), and `06c598c0` (LR, 0.702). The 2y_before_cv patient-split model and both frozen n=all models transfer best, suggesting that avoiding slice-level leakage during training and using all slices are complementary paths to external generalization.

2. **Full-slice frozen models transfer consistently well.** All four frozen n=all models rank in the top 4, with AUROC 0.635–0.718. Their frozen ViT-B/32 backbone produces transferable representations regardless of λ or split strategy.

3. **`6a1a1bdf` (previously rank 1 at 0.742) drops to rank 8 (0.615) after the resampling fix.** The old results were extracted without resampling the ablation MRIs; with correct 1×1×3 mm resampling the advantage disappears, suggesting its prior lead was an artefact of resolution mismatch.

4. **The 2y_before_cv gene set reversal**: slice-split `5d04e6ba` remains the worst embedding model (0.516) while patient-split sibling `9109a6c2` is the best (0.732). The patient-level split forces the model to learn more transferable image features rather than overfitting the resection gene-expression distribution.

5. **`050d401d` (bbox, slice-split) jumped from rank 12 to rank 6 (0.583→0.669) after the resampling fix.** The bbox pipeline crops around segmentation masks that were themselves resampled — getting this right mattered more for bbox than for raw-MRI models.

6. **Radiomic RF (0.590) sits mid-table**, ahead of 7 of the 16 embedding models.

7. **LR head dominates across the board after the fix.** LR wins for 13 of 16 models, including all frozen and most unfrozen configs.

8. **Freezing the backbone hurts bbox but helps raw MRI.** `92b9afed` (bbox, frozen, n=all, slice) achieves only 0.577 vs. unfrozen `050d401d` (0.669). For raw full-MRI, frozen + n=all adds +0.10 AUROC over unfrozen n=10; for bbox the same change costs −0.09.

### 5.2 Lausanne

1. **Overall performance is lower than on Soramic.** The top model (`1361bef2`, AUROC 0.771) matches Soramic's top, but the median drops significantly — 11 of 17 embedding models fall below 0.55, vs. only 5 on Soramic. The Lausanne cohort (74% positive rate) leaves little headroom for the downstream classifier to exploit specificity.

2. **`1361bef2` (λ=0.1, unfrozen, patient-split) is the clear top model at 0.771**, reversing its Soramic rank (rank 14 there at 0.522). Patient-level contrastive split appears to produce more transferable image features on Lausanne despite the different scanner/protocol.

3. **The frozen n=all group (Group 3) collapses completely.** All four frozen models rank 10–15 (AUROC 0.453–0.556), compared to ranks 2–4 on Soramic (0.684–0.718). The ViT-B/32 frozen backbone that transferred well from Soramic arterial phase does not transfer to the Lausanne `MRI_liver_arterial` acquisition.

4. **`9109a6c2` (2y_before_cv genes, patient-split) drops from Soramic rank 1 (0.732) to rank 7 (0.563).** The gene-expression-guided contrastive signal from the resection cohort does not generalize to Lausanne's different imaging protocol.

5. **`5d04e6ba` reversal holds but weakens.** On Soramic, the slice-split sibling (0.516) was worst while patient-split `9109a6c2` (0.732) was best. On Lausanne, `5d04e6ba` (0.655) now outperforms `9109a6c2` (0.563) — the pattern inverts.

6. **Bounding box models perform comparably to raw-MRI models**, unlike Soramic where bbox lagged. `92b9afed` (bbox, frozen, n=all) is rank 2 at 0.614, and `e12b0592` (bbox, λ=0.0, unfrozen, slice) reaches 0.595 at rank 6. The Lausanne masks appear reliable, making the bbox crop useful.

7. **Both radiomic baselines sit near chance (0.506–0.531)**, worse than on Soramic (0.518–0.590). The resection-trained radiomic pipeline does not generalise to the Lausanne acquisition.

8. **`dc7e1d10` (frozen, λ=0.1, slice-split) drops most dramatically**: Soramic rank 2 (0.718) → Lausanne rank 15 (0.453). Its resection CV (0.695, §4 rank 2) and Soramic transfer are both solid, yet it carries no Lausanne signal — likely overfitting slice-level patterns of the Soramic arterial protocol.

---

## 6. Ensemble results

Ensemble mode averages softmax probabilities from (a) the pre-trained radiomic pipeline and (b) an embedding head (SelectKBest k=100 + LR or RF, fitted on resection embeddings). The radiomic component is fixed to the model with the highest AUROC on each cohort:

- **Soramic**: radiomic RF (AUROC 0.590), `models/radiomics/radiomic_rfs_2year_rf.joblib`
- **Lausanne**: radiomic LR (AUROC 0.531), `models/radiomics/radiomic_rfs_2year_lr.joblib`

Multi-lesion strategy: average. Threshold: 0.5. Embedding AUROC column shows the best embedding-only head from Sections 2–3 for direct comparison.

### 6.1 Soramic — ensemble summary

Best ensemble head (LR or RF) per model, ranked by ensemble AUROC.

| Rank | Model ID | Head | Config | Emb AUROC | Ens AUROC | AUPRC | Sensitivity | Specificity | PPV | NPV | F1 |
|------|----------|------|--------|----------:|----------:|------:|------------:|------------:|----:|----:|---:|
| 1 | `dc7e1d10` | LR | raw, λ=0.1, frozen, n=all, slice | 0.718 | **0.765** | 0.880 | 0.829 | 0.389 | 0.725 | 0.538 | 0.773 |
| 2 | `a64b245f` | LR | raw, λ=0.0, frozen, n=all, slice | 0.684 | 0.735 | 0.865 | 0.971 | 0.278 | 0.723 | 0.833 | 0.829 |
| 3 | `9109a6c2` | LR | raw, λ=0.1, 2y_before_cv genes, n=10, patient | 0.732 | 0.733 | 0.844 | 0.829 | 0.389 | 0.725 | 0.538 | 0.773 |
| 4 | `12e4ba6a` | RF | raw, λ=0.1, predefined genes, n=10, slice | 0.670 | 0.729 | 0.853 | 0.857 | 0.444 | 0.750 | 0.615 | 0.800 |
| 5 | `06c598c0` | LR | raw, λ=0.0, frozen, n=all, patient | 0.702 | 0.717 | 0.846 | 0.914 | 0.222 | 0.696 | 0.571 | 0.790 |
| 6 | `5e3f71a0` | LR | raw, λ=0.1, frozen, n=all, patient | 0.635 | 0.681 | 0.835 | 0.971 | 0.000 | 0.654 | 0.000 | 0.782 |
| 7 | `6a1a1bdf` | LR | raw, λ=0.1, unfrozen, n=10, slice | 0.615 | 0.657 | 0.811 | 0.400 | 0.778 | 0.778 | 0.400 | 0.528 |
| 8 | `982a6fa2` | RF | raw, λ=0.0, unfrozen, n=10, slice | 0.606 | 0.618 | 0.740 | 0.971 | 0.167 | 0.694 | 0.750 | 0.810 |
| 9 | `a6f970d6` | LR | raw, λ=0.0, unfrozen, n=10, patient | 0.494 | 0.608 | 0.771 | 0.943 | 0.056 | 0.660 | 0.333 | 0.776 |
| — | radiomic RF | RF | 149 art. features, resection-trained | — | 0.590 | 0.766 | 0.143 | 1.000 | 1.000 | 0.375 | 0.250 |
| 10 | `1361bef2` | LR | raw, λ=0.1, unfrozen, n=10, patient | 0.522 | 0.598 | 0.762 | 1.000 | 0.056 | 0.673 | 1.000 | 0.805 |
| 11 | `5d04e6ba` | LR | raw, λ=0.1, 2y_before_cv genes, n=10, slice | 0.516 | 0.590 | 0.766 | 1.000 | 0.000 | 0.660 | — | 0.795 |
| 12 | `92b9afed` | RF | bbox, λ=0.1, frozen, n=all, slice | 0.577 | 0.590 | 0.763 | 0.857 | 0.056 | 0.638 | 0.167 | 0.732 |
| 13 | `f8aabb75` | LR | bbox, λ=0.1, unfrozen, n=10, patient | 0.539 | 0.587 | 0.765 | 0.943 | 0.056 | 0.660 | 0.333 | 0.776 |
| 14 | `050d401d` | LR | bbox, λ=0.1, unfrozen, n=10, slice | 0.669 | 0.583 | 0.761 | 1.000 | 0.000 | 0.660 | — | 0.795 |
| 15 | `34e6806f` | LR | raw, λ=0.1, predefined genes, n=10, patient | 0.574 | 0.571 | 0.704 | 0.657 | 0.500 | 0.719 | 0.429 | 0.687 |
| 16 | `8715461c` | LR | bbox, λ=0.0, unfrozen, n=10, patient | 0.534 | 0.548 | 0.688 | 0.029 | 0.889 | 0.333 | 0.320 | 0.053 |
| 17 | `e12b0592` | LR | bbox, λ=0.0, unfrozen, n=10, slice | 0.517 | 0.533 | 0.724 | 0.457 | 0.556 | 0.667 | 0.345 | 0.542 |

### 6.2 Lausanne — ensemble summary

Best ensemble head (LR or RF) per model, ranked by ensemble AUROC.

| Rank | Model ID | Head | Config | Emb AUROC | Ens AUROC | AUPRC | Sensitivity | Specificity | PPV | NPV | F1 |
|------|----------|------|--------|----------:|----------:|------:|------------:|------------:|----:|----:|---:|
| 1 | `1361bef2` | RF | raw, λ=0.1, unfrozen, n=10, patient | 0.771 | **0.653** | 0.839 | 0.422 | 0.625 | 0.760 | 0.278 | 0.543 |
| 2 | `5d04e6ba` | LR | raw, λ=0.1, 2y_before_cv genes, n=10, slice | 0.655 | 0.592 | 0.805 | 0.600 | 0.625 | 0.818 | 0.357 | 0.692 |
| 3 | `a6f970d6` | LR | raw, λ=0.0, unfrozen, n=10, patient | 0.618 | 0.585 | 0.798 | 0.422 | 0.625 | 0.760 | 0.278 | 0.543 |
| 4 | `92b9afed` | LR | bbox, λ=0.1, frozen, n=all, slice | 0.614 | 0.578 | 0.817 | 0.422 | 0.625 | 0.760 | 0.278 | 0.543 |
| 5 | `982a6fa2` | LR | raw, λ=0.0, unfrozen, n=10, slice | 0.600 | 0.575 | 0.796 | 0.422 | 0.625 | 0.760 | 0.278 | 0.543 |
| 6 | `e12b0592` | RF | bbox, λ=0.0, unfrozen, n=10, slice | 0.595 | 0.556 | 0.790 | 0.422 | 0.625 | 0.760 | 0.278 | 0.543 |
| 7 | `5e3f71a0` | LR | raw, λ=0.1, frozen, n=all, patient | 0.534 | 0.554 | 0.784 | 0.422 | 0.625 | 0.760 | 0.278 | 0.543 |
| 8 | `9109a6c2` | LR | raw, λ=0.1, 2y_before_cv genes, n=10, patient | 0.563 | 0.550 | 0.797 | 0.422 | 0.625 | 0.760 | 0.278 | 0.543 |
| 9 | `a64b245f` | RF | raw, λ=0.0, frozen, n=all, slice | 0.556 | 0.542 | 0.817 | 0.422 | 0.625 | 0.760 | 0.278 | 0.543 |
| — | radiomic LR | LR | 149 art. features, resection-trained | — | 0.531 | 0.748 | 0.422 | 0.625 | 0.760 | 0.278 | 0.543 |
| 10 | `6a1a1bdf` | LR | raw, λ=0.1, unfrozen, n=10, slice | 0.497 | 0.538 | 0.767 | 0.422 | 0.625 | 0.760 | 0.278 | 0.543 |
| 11 | `06c598c0` | LR | raw, λ=0.0, frozen, n=all, patient | 0.515 | 0.536 | 0.794 | 0.422 | 0.625 | 0.760 | 0.278 | 0.543 |
| 12 | `050d401d` | RF | bbox, λ=0.1, unfrozen, n=10, slice | 0.544 | 0.534 | 0.747 | 0.422 | 0.625 | 0.760 | 0.278 | 0.543 |
| 13 | `f8aabb75` | RF | bbox, λ=0.1, unfrozen, n=10, patient | 0.515 | 0.530 | 0.793 | 0.422 | 0.625 | 0.760 | 0.278 | 0.543 |
| 14 | `8715461c` | LR | bbox, λ=0.0, unfrozen, n=10, patient | 0.494 | 0.519 | 0.753 | 0.422 | 0.625 | 0.760 | 0.278 | 0.543 |
| 15 | `dc7e1d10` | RF | raw, λ=0.1, frozen, n=all, slice | 0.453 | 0.504 | 0.784 | 0.422 | 0.625 | 0.760 | 0.278 | 0.543 |
| 16 | `12e4ba6a` | RF | raw, λ=0.1, predefined genes, n=10, slice | 0.477 | 0.497 | 0.762 | 0.422 | 0.625 | 0.760 | 0.278 | 0.543 |
| 17 | `34e6806f` | LR | raw, λ=0.1, predefined genes, n=10, patient | 0.420 | 0.478 | 0.714 | 0.422 | 0.625 | 0.760 | 0.278 | 0.543 |

---

## 7. File references

| Artifact | Path |
|---|---|
| Radiomic LR (Soramic) | `results/eval/ablation/radiomic_lr_rfs_2year_{timestamp}.json` |
| Radiomic RF (Soramic) | `results/eval/ablation/radiomic_rf_rfs_2year_{timestamp}.json` |
| Embedding results — Soramic | `results/eval/ablation/embedding_{model_id}_rfs_2year_{timestamp}.json` |
| Radiomic LR/RF (Lausanne) | `results/eval/lusanne/radiomic_rfs_2year_{timestamp}.json` |
| Embedding results — Lausanne | `results/eval/lusanne/embedding_{model_id}_rfs_2year_{timestamp}.json` |
| Radiomic models | `models/radiomics/radiomic_rfs_2year_{lr,rf}.joblib` |
| Contrastive models | `training/contrastive/{model_id}/best_model.pt` |
| Cached Soramic embeddings | `training/contrastive/{model_id}/cached_embeddings/ablation_img_emb_{raw,bbox}.parquet` |
| Cached Lausanne embeddings | `training/contrastive/{model_id}/cached_embeddings/ablation_lusanne_img_emb_{raw,bbox}.parquet` |
