# Ablation Cohort Evaluation — 2-Year RFS Prediction (v4)
**Date:** 2026-07-27  
**Covers:** Soramic (ablation) cohort · Lausanne cohort  
**Based on:** `reports/0713/0713_ablation_eval_v3.md`  
**Change vs 0713 v3:** all model configurations trained with **n=10 slices per patient are dropped**. Only the frozen, all-slice (**n=all**) configurations remain. Groups 1 and 2 are removed in full; all tables are re-ranked over the surviving models.

**Update 2026-07-21:** the bbox frozen n=all family is completed by adding three configurations trained via `scripts/submit_bbox_frozen_train.sh` — `16acfdd9` (λ=0.0, slice), `3baefc68` (λ=0.0, patient), `8fcb5dd3` (λ=0.1, patient) — so Group 4 now mirrors the raw Group 3's λ × split grid. **8 models total** (4 raw Group 3 + 4 bbox Group 4). All tables (§2–§6) are re-ranked over the 8.

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

8 models evaluated on both cohorts (all frozen, n=all). Downstream head: SelectKBest(f_classif, k=100) + LR or RF fitted on resection embeddings, evaluated on test cohort embeddings. MRI: arterial phase, mean-pooled sagittal slices. Embeddings resampled to 1×1×3 mm before extraction.

#### 1.3.1 Contrastive training configurations

All remaining models share the same ViT-B/32 backbone and triplet contrastive objective, use a **frozen** backbone, and are trained on **all** available slices per patient (**n=all**). The n=10 and unfrozen configurations from the 0713 v3 report are dropped. The axes that still vary across these 8 models are:

| Parameter | Values | Notes |
|---|---|---|
| **λ** | 0.0, 0.1 | Weight of the contrastive loss relative to the supervised classification loss. λ=0.0 trains without contrastive signal. |
| **Input** | raw, bbox | Raw full MRI vs. a bounding-box crop around the segmentation mask. |
| **Validation split** | slice, patient | Unit of train/val split during training. Slice-level split may allow the same patient's slices to appear in both train and val; patient-level split holds out full patients. |

The 8 model configurations (all frozen, n=all, 40-gene set):

| Group | Model ID | Input | Gene set | λ | Frozen | n slices | Val split |
|-------|----------|-------|----------|---|--------|----------|-----------|
| 3 | `dc7e1d10` | raw | 40 genes | 0.1 | yes | all | slice |
| 3 | `5e3f71a0` | raw | 40 genes | 0.1 | yes | all | patient |
| 3 | `a64b245f` | raw | 40 genes | 0.0 | yes | all | slice |
| 3 | `06c598c0` | raw | 40 genes | 0.0 | yes | all | patient |
| 4 | `92b9afed` | bbox | 40 genes | 0.1 | yes | all | slice |
| 4 | `8fcb5dd3` | bbox | 40 genes | 0.1 | yes | all | patient |
| 4 | `16acfdd9` | bbox | 40 genes | 0.0 | yes | all | slice |
| 4 | `3baefc68` | bbox | 40 genes | 0.0 | yes | all | patient |

Group 3: frozen backbone with all slices (raw MRI); Group 4: bounding-box crop input (frozen, all slices). Both groups now span the full λ ∈ {0.0, 0.1} × split ∈ {slice, patient} grid (`8fcb5dd3`/`16acfdd9`/`3baefc68` added 2026-07-21).

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
| λ=0.1, frozen, n=all, slice | `92b9afed` | 0.571 | 0.577 | 0.577 |
| λ=0.1, frozen, n=all, patient | `8fcb5dd3` | 0.551 | 0.629 | 0.629 |
| λ=0.0, frozen, n=all, slice | `16acfdd9` | 0.480 | 0.395 | 0.480 |
| λ=0.0, frozen, n=all, patient | `3baefc68` | 0.531 | 0.449 | 0.531 |

### 2.3 Summary table

Best head (LR or RF, whichever has higher AUROC) per model, ranked by AUROC. Multi-lesion: average. Threshold: 0.5. "—" = undefined NPV (all samples predicted positive).

| Rank | Model ID | Head | Config | AUROC | AUPRC | Sensitivity | Specificity | PPV | NPV | F1 |
|------|----------|------|--------|------:|------:|------------:|------------:|----:|----:|---:|
| 1 | `dc7e1d10` | LR | raw, λ=0.1, frozen, n=all, slice | **0.718** | 0.838 | 0.846 | 0.389 | 0.750 | 0.538 | 0.795 |
| 2 | `06c598c0` | LR | raw, λ=0.0, frozen, n=all, patient | 0.702 | 0.840 | 0.897 | 0.222 | 0.714 | 0.500 | 0.795 |
| 3 | `a64b245f` | LR | raw, λ=0.0, frozen, n=all, slice | 0.684 | 0.804 | 0.974 | 0.278 | 0.745 | 0.833 | 0.844 |
| 4 | `5e3f71a0` | RF | raw, λ=0.1, frozen, n=all, patient | 0.635 | 0.819 | 1.000 | 0.000 | 0.684 | — | 0.812 |
| 5 | `8fcb5dd3` | RF | bbox, λ=0.1, frozen, n=all, patient | 0.629 | 0.792 | 1.000 | 0.000 | 0.679 | — | 0.809 |
| — | radiomic RF | RF | 149 art. features, resection-trained | 0.590 | 0.766 | 0.143 | 1.000 | 1.000 | 0.375 | 0.250 |
| 6 | `92b9afed` | RF | bbox, λ=0.1, frozen, n=all, slice | 0.577 | 0.719 | 0.972 | 0.000 | 0.660 | 0.000 | 0.787 |
| 7 | `3baefc68` | LR | bbox, λ=0.0, frozen, n=all, patient | 0.531 | 0.710 | 0.947 | 0.111 | 0.692 | 0.500 | 0.800 |
| — | radiomic LR | LR | 149 art. features, resection-trained | 0.518 | 0.671 | 0.657 | 0.389 | 0.676 | 0.368 | 0.667 |
| 8 | `16acfdd9` | LR | bbox, λ=0.0, frozen, n=all, slice | 0.480 | 0.755 | 0.395 | 0.667 | 0.714 | 0.343 | 0.508 |

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
| λ=0.1, frozen, n=all, slice | `92b9afed` | 0.607 | 0.614 | **0.614** |
| λ=0.1, frozen, n=all, patient | `8fcb5dd3` | 0.434 | 0.588 | 0.588 |
| λ=0.0, frozen, n=all, slice | `16acfdd9` | 0.381 | 0.396 | 0.396 |
| λ=0.0, frozen, n=all, patient | `3baefc68` | 0.487 | 0.399 | 0.487 |

### 3.3 Summary table

Best head (LR or RF, whichever has higher AUROC) per model, ranked by AUROC. Multi-lesion: average. Threshold: 0.5. "—" = undefined NPV (all samples predicted positive).

| Rank | Model ID | Head | Config | AUROC | AUPRC | Sensitivity | Specificity | PPV | NPV | F1 |
|------|----------|------|--------|------:|------:|------------:|------------:|----:|----:|---:|
| 1 | `92b9afed` | RF | bbox, λ=0.1, frozen, n=all, slice | **0.614** | 0.810 | 0.979 | 0.059 | 0.746 | 0.500 | 0.847 |
| 2 | `8fcb5dd3` | RF | bbox, λ=0.1, frozen, n=all, patient | 0.588 | 0.791 | 1.000 | 0.000 | 0.738 | — | 0.850 |
| 3 | `a64b245f` | RF | raw, λ=0.0, frozen, n=all, slice | 0.556 | 0.803 | 0.959 | 0.059 | 0.746 | 0.333 | 0.839 |
| 4 | `5e3f71a0` | LR | raw, λ=0.1, frozen, n=all, patient | 0.534 | 0.775 | 0.939 | 0.118 | 0.754 | 0.400 | 0.836 |
| — | radiomic LR | LR | 149 art. features, resection-trained | 0.531 | 0.748 | 0.422 | 0.625 | 0.760 | 0.278 | 0.543 |
| 5 | `06c598c0` | LR | raw, λ=0.0, frozen, n=all, patient | 0.515 | 0.779 | 0.755 | 0.294 | 0.755 | 0.294 | 0.755 |
| — | radiomic RF | RF | 149 art. features, resection-trained | 0.506 | 0.747 | 0.067 | 0.875 | 0.600 | 0.250 | 0.120 |
| 6 | `3baefc68` | LR | bbox, λ=0.0, frozen, n=all, patient | 0.487 | 0.742 | 0.812 | 0.176 | 0.736 | 0.250 | 0.772 |
| 7 | `dc7e1d10` | RF | raw, λ=0.1, frozen, n=all, slice | 0.453 | 0.727 | 0.776 | 0.118 | 0.717 | 0.154 | 0.745 |
| 8 | `16acfdd9` | RF | bbox, λ=0.0, frozen, n=all, slice | 0.396 | 0.696 | 0.979 | 0.000 | 0.734 | 0.000 | 0.839 |

---

## 4. Rank by CV AUC

In-CV AUC uses **fixed LR/RF, no hyperparameter search, no feature selection**, all on the **same
image extraction as the transfer columns** — the survival `resection_img_emb.parquet` cache (128-dim
image-only, patient-level mean-pooled). LR/RF are taken as-is from `hcc_multimodal/baselines/config.py`
(`MODELS`; LR = saga, elasticnet, l1_ratio=1.0, C=1.0; RF = 100 trees, default depth) and fit on
**all 128 image dims** through `SimpleImputer(median) → StandardScaler → classifier` in a plain
**3-fold** stratified CV on the 54-patient (26 positive) resection cohort. "Best head" = higher mean
fold AUC. Soramic/Lausanne columns are the **CV-selected head's own** transfer — that same head refit
on all resection and applied to each cohort through the identical all-128 no-FS pipeline (not the
§2–3 SelectKBest(k=100), best-of-{LR,RF} numbers, which mix a different head per cohort). Δ vs.
best radiomic baseline (Soramic RF=0.590; Lausanne LR=0.531).

> **Note — one extraction throughout.** Earlier drafts computed this CV on the `multimodal_prediction`
> `emb_*.parquet` cache, a *different* extraction from the §2–3 transfer cache (per-patient vectors
> only ~0.77 cosine apart), which inflated the most-separable slice-split models to CV≈1.000. CV,
> transfer, and the companion §6 grid now all read `resection_img_emb.parquet`, so the columns are
> directly comparable and the spurious 1.000s are gone (`dc7e1d10` → 0.695).

> **Convergence note (2026-07-20).** The LR head's `max_iter` was bumped 1000 → 5000 so the saga
> solver converges on the 128-dim embedding. This lowers a few LR CV values slightly (`dc7e1d10`
> 0.699 → **0.695**) and flips two best-heads at convergence (`92b9afed` RF→LR, `5e3f71a0` LR→RF).
> `dc7e1d10` 0.695 now matches the `LASSO`/`All features` grid anchor in
> `reports/0720/0720_embedding_grid_eval_v3.md`.
>
> **Transfer-column correction (2026-07-20).** The Soramic/Lausanne columns previously reused the
> §2–3 **best-of-{LR,RF}** transfer numbers, which pick a *different* head per cohort — for
> `dc7e1d10` that meant Soramic from LR (0.718) but Lausanne from RF (0.453), so the row silently
> mixed heads and the Lausanne cell disagreed with the LR-only grid anchor (0.419). Both columns now
> report the **CV-selected head's** all-128 transfer, so the row is single-head throughout. `dc7e1d10`
> is unchanged on Soramic (LR won both, 0.718) and corrects to **Lausanne 0.419**, matching the grid
> anchor; Soramic shifts for rows where the §2–3 best-of previously used RF or SelectKBest(k=100).

| CV Rank | Model ID | Config | Best head | CV AUC ± std | Soramic AUROC | Lausanne AUROC |
|--------:|----------|--------|-----------|-------------:|--------------:|---------------:|
| 1 | `dc7e1d10` | raw, λ=0.1, frozen, n=all, slice | LR | **0.695 ± 0.117** | **0.718 (+0.128)** | 0.419 (−0.112) |
| 2 | `a64b245f` | raw, λ=0.0, frozen, n=all, slice | LR | 0.665 ± 0.092 | 0.688 (+0.098) | 0.455 (−0.076) |
| 3 | `92b9afed` | bbox, λ=0.1, frozen, n=all, slice | LR | 0.662 ± 0.063 | 0.554 (−0.036) | **0.591 (+0.060)** |
| 4 | `5e3f71a0` | raw, λ=0.1, frozen, n=all, patient | RF | 0.616 ± 0.010 | 0.594 (+0.004) | 0.534 (+0.003) |
| 5 | `06c598c0` | raw, λ=0.0, frozen, n=all, patient | LR | 0.603 ± 0.160 | 0.717 (+0.127) | 0.505 (−0.026) |
| 6 | `8fcb5dd3` | bbox, λ=0.1, frozen, n=all, patient | RF | 0.579 ± 0.025 | 0.451 (−0.139) | 0.387 (−0.144) |
| 7 | `3baefc68` | bbox, λ=0.0, frozen, n=all, patient | RF | 0.568 ± 0.067 | 0.447 (−0.143) | 0.525 (−0.006) |
| 8 | `16acfdd9` | bbox, λ=0.0, frozen, n=all, slice | RF | 0.552 ± 0.098 | 0.601 (+0.011) | 0.425 (−0.106) |

**Even within the frozen n=all family, resection CV does not cleanly point to the best-transferring
embedding.** The CV-top `dc7e1d10` (0.695) does transfer on Soramic (0.718, +0.128) but is the worst
of the raw models on Lausanne (0.419, −0.112). `a64b245f` (CV rank 2) shows the same shape — strong
Soramic (0.688), weak Lausanne (0.455). The one model that leads on Lausanne, `92b9afed` (bbox, 0.591),
sits mid-pack on CV (rank 3) and near-worst on Soramic (0.554). The two cohorts continue to pull in
opposite directions: the raw-frozen models that top Soramic are bottom on Lausanne, and vice versa.

**The three new bbox patient/slice configs (`8fcb5dd3`, `3baefc68`, `16acfdd9`) fill the bottom of the
CV ranking (0.552–0.579), all RF-headed and below `92b9afed`.** Their all-128 transfer is mostly at or
below chance — the exception is `16acfdd9` (bbox, λ=0.0, slice), the only new model above the radiomic
Soramic baseline (0.601, +0.011) despite the lowest CV of the eight (0.552), reprising the same
CV↔transfer disconnect seen in the raw group. `3baefc68` is roughly at the radiomic Lausanne baseline
(0.525, −0.006). A per-head FS × classifier grid on `dc7e1d10` is in the companion
`reports/0713/0713_embedding_grid_eval_v2.md` §6.

---

## 5. Observations

### 5.1 Soramic

1. **The top three models (AUROC 0.68–0.72) are all raw frozen n=all configs**: `dc7e1d10` (LR, 0.718), `06c598c0` (LR, 0.702), and `a64b245f` (LR, 0.684). Using all slices with a frozen backbone is the path to external generalization on Soramic.

2. **All four raw frozen n=all models rank in the top 4** (AUROC 0.635–0.718). Their frozen ViT-B/32 backbone produces transferable representations regardless of λ or split strategy.

3. **The new bbox `8fcb5dd3` (λ=0.1, patient) is the best bbox model on Soramic (RF, 0.629), rank 5** — it slots between the raw frozen group and the radiomic RF baseline, and is the only bbox config to beat radiomic RF (0.590). The remaining bbox configs trail: `92b9afed` 0.577, `3baefc68` 0.531, `16acfdd9` 0.480 (last).

4. **Within the bbox family, patient-split and λ=0.1 both help.** The two λ=0.1 bbox models (`8fcb5dd3` patient 0.629, `92b9afed` slice 0.577) beat the two λ=0.0 bbox models (`3baefc68` patient 0.531, `16acfdd9` slice 0.480) — the reverse of the raw group, where λ barely mattered.

5. **Radiomic RF (0.590) sits mid-table**, above three of the four bbox models (`92b9afed`, `3baefc68`, `16acfdd9`) but below all raw frozen models and the best bbox model `8fcb5dd3`.

6. **Head choice is mixed.** LR wins for the three top raw configs and both bbox λ=0.0 models (`3baefc68`, `16acfdd9`); RF wins for `5e3f71a0`, `92b9afed`, and `8fcb5dd3`.

### 5.2 Lausanne

1. **Overall performance is lower than on Soramic, and the top two are now both bbox.** `92b9afed` (RF, 0.614) leads and the new `8fcb5dd3` (RF, 0.588) is second — both λ=0.1 bbox models — while six of eight embedding models fall at or below 0.556.

2. **The raw frozen group that topped Soramic collapses on Lausanne.** All four raw frozen models rank 5–7 (AUROC 0.453–0.556), compared to the top 4 on Soramic (0.635–0.718). The frozen ViT-B/32 backbone that transferred well from Soramic arterial phase does not transfer to the Lausanne `MRI_liver_arterial` acquisition.

3. **The bbox crop helps Lausanne only at λ=0.1.** The two λ=0.1 bbox models top the table (0.588–0.614), but the two λ=0.0 bbox models are weak — `3baefc68` (0.487) and `16acfdd9` (0.396, worst overall). So the bbox advantage on Lausanne is contingent on the contrastive signal, not the crop alone.

4. **`16acfdd9` (bbox, λ=0.0, slice) is the worst model on Lausanne (0.396), and `dc7e1d10` the worst raw (0.453)** — its Soramic rank 1 (0.718) → Lausanne rank 7 remains the sharpest cross-cohort reversal, consistent with overfitting slice-level Soramic-protocol patterns.

5. **Both radiomic baselines sit near chance (0.506–0.531)**, worse than on Soramic (0.518–0.590), and now outrank three of the eight embedding models. The resection-trained radiomic pipeline does not generalise to the Lausanne acquisition either.

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
| 3 | `06c598c0` | LR | raw, λ=0.0, frozen, n=all, patient | 0.702 | 0.717 | 0.846 | 0.914 | 0.222 | 0.696 | 0.571 | 0.790 |
| 4 | `5e3f71a0` | LR | raw, λ=0.1, frozen, n=all, patient | 0.635 | 0.681 | 0.835 | 0.971 | 0.000 | 0.654 | 0.000 | 0.782 |
| 5 | `8fcb5dd3` | LR | bbox, λ=0.1, frozen, n=all, patient | 0.629 | 0.660 | 0.797 | 1.000 | 0.056 | 0.673 | 1.000 | 0.805 |
| — | radiomic RF | RF | 149 art. features, resection-trained | — | 0.590 | 0.766 | 0.143 | 1.000 | 1.000 | 0.375 | 0.250 |
| 6 | `92b9afed` | RF | bbox, λ=0.1, frozen, n=all, slice | 0.577 | 0.590 | 0.763 | 0.857 | 0.056 | 0.638 | 0.167 | 0.732 |
| 7 | `3baefc68` | LR | bbox, λ=0.0, frozen, n=all, patient | 0.531 | 0.575 | 0.748 | 0.971 | 0.111 | 0.680 | 0.667 | 0.800 |
| 8 | `16acfdd9` | LR | bbox, λ=0.0, frozen, n=all, slice | 0.480 | 0.544 | 0.765 | 0.400 | 0.667 | 0.700 | 0.364 | 0.509 |

### 6.2 Lausanne — ensemble summary

Best ensemble head (LR or RF) per model, ranked by ensemble AUROC.

| Rank | Model ID | Head | Config | Emb AUROC | Ens AUROC | AUPRC | Sensitivity | Specificity | PPV | NPV | F1 |
|------|----------|------|--------|----------:|----------:|------:|------------:|------------:|----:|----:|---:|
| 1 | `92b9afed` | LR | bbox, λ=0.1, frozen, n=all, slice | 0.614 | **0.578** | 0.817 | 0.422 | 0.625 | 0.760 | 0.278 | 0.543 |
| 2 | `8fcb5dd3` | RF | bbox, λ=0.1, frozen, n=all, patient | 0.588 | 0.576 | 0.761 | 0.422 | 0.625 | 0.760 | 0.278 | 0.543 |
| 3 | `5e3f71a0` | LR | raw, λ=0.1, frozen, n=all, patient | 0.534 | 0.554 | 0.784 | 0.422 | 0.625 | 0.760 | 0.278 | 0.543 |
| 4 | `a64b245f` | RF | raw, λ=0.0, frozen, n=all, slice | 0.556 | 0.542 | 0.817 | 0.422 | 0.625 | 0.760 | 0.278 | 0.543 |
| 5 | `06c598c0` | LR | raw, λ=0.0, frozen, n=all, patient | 0.515 | 0.536 | 0.794 | 0.422 | 0.625 | 0.760 | 0.278 | 0.543 |
| — | radiomic LR | LR | 149 art. features, resection-trained | — | 0.531 | 0.748 | 0.422 | 0.625 | 0.760 | 0.278 | 0.543 |
| 6 | `3baefc68` | LR | bbox, λ=0.0, frozen, n=all, patient | 0.487 | 0.518 | 0.759 | 0.422 | 0.625 | 0.760 | 0.278 | 0.543 |
| 7 | `dc7e1d10` | RF | raw, λ=0.1, frozen, n=all, slice | 0.453 | 0.504 | 0.784 | 0.422 | 0.625 | 0.760 | 0.278 | 0.543 |
| 8 | `16acfdd9` | RF | bbox, λ=0.0, frozen, n=all, slice | 0.396 | 0.478 | 0.719 | 0.422 | 0.625 | 0.760 | 0.278 | 0.543 |

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
