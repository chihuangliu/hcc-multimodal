# Ablation Cohort Evaluation — 2-Year RFS Prediction (v3)
**Date:** 2026-07-27  
**Covers:** Soramic (ablation) cohort · Lausanne cohort  
**Based on:** `reports/0713/0713_ablation_eval_v3.md`  
**Change vs 0713 v3:** all model configurations trained with **n=10 slices per patient are dropped**. Only the frozen, all-slice (**n=all**) configurations remain — 5 models total (4 raw Group 3 + 1 bbox Group 4). Groups 1 and 2 are removed in full; all tables are re-ranked over the surviving models.

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

5 models evaluated on both cohorts (all frozen, n=all). Downstream head: SelectKBest(f_classif, k=100) + LR or RF fitted on resection embeddings, evaluated on test cohort embeddings. MRI: arterial phase, mean-pooled sagittal slices. Embeddings resampled to 1×1×3 mm before extraction.

#### 1.3.1 Contrastive training configurations

All remaining models share the same ViT-B/32 backbone and triplet contrastive objective, use a **frozen** backbone, and are trained on **all** available slices per patient (**n=all**). The n=10 and unfrozen configurations from the 0713 v3 report are dropped. The axes that still vary across these 5 models are:

| Parameter | Values | Notes |
|---|---|---|
| **λ** | 0.0, 0.1 | Weight of the contrastive loss relative to the supervised classification loss. λ=0.0 trains without contrastive signal. |
| **Input** | raw, bbox | Raw full MRI vs. a bounding-box crop around the segmentation mask. |
| **Validation split** | slice, patient | Unit of train/val split during training. Slice-level split may allow the same patient's slices to appear in both train and val; patient-level split holds out full patients. |

The 5 surviving model configurations (all frozen, n=all, 40-gene set):

| Group | Model ID | Input | Gene set | λ | Frozen | n slices | Val split |
|-------|----------|-------|----------|---|--------|----------|-----------|
| 3 | `dc7e1d10` | raw | 40 genes | 0.1 | yes | all | slice |
| 3 | `5e3f71a0` | raw | 40 genes | 0.1 | yes | all | patient |
| 3 | `a64b245f` | raw | 40 genes | 0.0 | yes | all | slice |
| 3 | `06c598c0` | raw | 40 genes | 0.0 | yes | all | patient |
| 4 | `92b9afed` | bbox | 40 genes | 0.1 | yes | all | slice |

Group 3: frozen backbone with all slices (raw MRI); Group 4: bounding-box crop input (frozen, all slices).

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

### 2.3 Summary table

Best head (LR or RF, whichever has higher AUROC) per model, ranked by AUROC. Multi-lesion: average. Threshold: 0.5. "—" = undefined NPV (all samples predicted positive).

| Rank | Model ID | Head | Config | AUROC | AUPRC | Sensitivity | Specificity | PPV | NPV | F1 |
|------|----------|------|--------|------:|------:|------------:|------------:|----:|----:|---:|
| 1 | `dc7e1d10` | LR | raw, λ=0.1, frozen, n=all, slice | **0.718** | 0.838 | 0.846 | 0.389 | 0.750 | 0.538 | 0.795 |
| 2 | `06c598c0` | LR | raw, λ=0.0, frozen, n=all, patient | 0.702 | 0.840 | 0.897 | 0.222 | 0.714 | 0.500 | 0.795 |
| 3 | `a64b245f` | LR | raw, λ=0.0, frozen, n=all, slice | 0.684 | 0.804 | 0.974 | 0.278 | 0.745 | 0.833 | 0.844 |
| 4 | `5e3f71a0` | RF | raw, λ=0.1, frozen, n=all, patient | 0.635 | 0.819 | 1.000 | 0.000 | 0.684 | — | 0.812 |
| — | radiomic RF | RF | 149 art. features, resection-trained | 0.590 | 0.766 | 0.143 | 1.000 | 1.000 | 0.375 | 0.250 |
| 5 | `92b9afed` | RF | bbox, λ=0.1, frozen, n=all, slice | 0.577 | 0.719 | 0.972 | 0.000 | 0.660 | 0.000 | 0.787 |
| — | radiomic LR | LR | 149 art. features, resection-trained | 0.518 | 0.671 | 0.657 | 0.389 | 0.676 | 0.368 | 0.667 |

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

### 3.3 Summary table

Best head (LR or RF, whichever has higher AUROC) per model, ranked by AUROC. Multi-lesion: average. Threshold: 0.5. "—" = undefined NPV (all samples predicted positive).

| Rank | Model ID | Head | Config | AUROC | AUPRC | Sensitivity | Specificity | PPV | NPV | F1 |
|------|----------|------|--------|------:|------:|------------:|------------:|----:|----:|---:|
| 1 | `92b9afed` | RF | bbox, λ=0.1, frozen, n=all, slice | **0.614** | 0.810 | 0.979 | 0.059 | 0.746 | 0.500 | 0.847 |
| 2 | `a64b245f` | RF | raw, λ=0.0, frozen, n=all, slice | 0.556 | 0.803 | 0.959 | 0.059 | 0.746 | 0.333 | 0.839 |
| 3 | `5e3f71a0` | LR | raw, λ=0.1, frozen, n=all, patient | 0.534 | 0.775 | 0.939 | 0.118 | 0.754 | 0.400 | 0.836 |
| — | radiomic LR | LR | 149 art. features, resection-trained | 0.531 | 0.748 | 0.422 | 0.625 | 0.760 | 0.278 | 0.543 |
| 4 | `06c598c0` | LR | raw, λ=0.0, frozen, n=all, patient | 0.515 | 0.779 | 0.755 | 0.294 | 0.755 | 0.294 | 0.755 |
| — | radiomic RF | RF | 149 art. features, resection-trained | 0.506 | 0.747 | 0.067 | 0.875 | 0.600 | 0.250 | 0.120 |
| 5 | `dc7e1d10` | RF | raw, λ=0.1, frozen, n=all, slice | 0.453 | 0.727 | 0.776 | 0.118 | 0.717 | 0.154 | 0.745 |

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

**Even within the frozen n=all family, resection CV does not cleanly point to the best-transferring
embedding.** The CV-top `dc7e1d10` (0.695) does transfer on Soramic (0.718, +0.128) but is the worst
of the five on Lausanne (0.419, −0.112). `a64b245f` (CV rank 2) shows the same shape — strong Soramic
(0.688), weak Lausanne (0.455). The one model that leads on Lausanne, `92b9afed` (bbox, 0.591), sits
mid-pack on CV (rank 3) and near-worst on Soramic (0.554). The two cohorts continue to pull in
opposite directions: the raw-frozen models that top Soramic are bottom on Lausanne, and vice versa. A
per-head FS × classifier grid on `dc7e1d10` is in the companion
`reports/0713/0713_embedding_grid_eval_v2.md` §6.

---

## 5. Observations

### 5.1 Soramic

1. **The top three models (AUROC 0.68–0.72) are all raw frozen n=all configs**: `dc7e1d10` (LR, 0.718), `06c598c0` (LR, 0.702), and `a64b245f` (LR, 0.684). Using all slices with a frozen backbone is the path to external generalization on Soramic.

2. **All four raw frozen n=all models rank in the top 4** (AUROC 0.635–0.718). Their frozen ViT-B/32 backbone produces transferable representations regardless of λ or split strategy.

3. **λ has no clear effect within the frozen family.** λ=0.1 slice (`dc7e1d10`, 0.718) edges out λ=0.0 configs (`06c598c0` 0.702, `a64b245f` 0.684), but the spread is small and both λ settings land in the top 4.

4. **Radiomic RF (0.590) sits just above the bbox model.** It outranks only `92b9afed` (0.577) among the surviving embedding models; all four raw frozen models beat it.

5. **LR head dominates.** LR wins for the three top raw configs (`dc7e1d10`, `06c598c0`, `a64b245f`); RF wins only for `5e3f71a0` and the bbox `92b9afed`.

6. **Freezing hurts bbox relative to raw MRI.** `92b9afed` (bbox, frozen, n=all, slice) reaches only 0.577 — the lowest of the five — versus 0.635–0.718 for the raw frozen models on the same frozen backbone.

### 5.2 Lausanne

1. **Overall performance is lower than on Soramic.** The best surviving model (`92b9afed`, AUROC 0.614) trails Soramic's top (0.718), and four of five embedding models fall at or below 0.556.

2. **The raw frozen group that topped Soramic collapses on Lausanne.** All four raw frozen models rank in the bottom tier (AUROC 0.453–0.556), compared to the top 4 on Soramic (0.635–0.718). The frozen ViT-B/32 backbone that transferred well from Soramic arterial phase does not transfer to the Lausanne `MRI_liver_arterial` acquisition.

3. **`dc7e1d10` drops most dramatically**: Soramic rank 1 (0.718) → Lausanne last (0.453). Its resection CV (0.695, §4 rank 1) and Soramic transfer are both solid, yet it carries no Lausanne signal — likely overfitting slice-level patterns of the Soramic arterial protocol.

4. **The bbox model reverses.** `92b9afed` is last on Soramic (0.577) but tops Lausanne (0.614). The Lausanne masks appear reliable, making the bbox crop useful where raw-MRI frozen features do not transfer.

5. **Both radiomic baselines sit near chance (0.506–0.531)**, worse than on Soramic (0.518–0.590), and now outrank two of the five embedding models. The resection-trained radiomic pipeline does not generalise to the Lausanne acquisition either.

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
| — | radiomic RF | RF | 149 art. features, resection-trained | — | 0.590 | 0.766 | 0.143 | 1.000 | 1.000 | 0.375 | 0.250 |
| 5 | `92b9afed` | RF | bbox, λ=0.1, frozen, n=all, slice | 0.577 | 0.590 | 0.763 | 0.857 | 0.056 | 0.638 | 0.167 | 0.732 |

### 6.2 Lausanne — ensemble summary

Best ensemble head (LR or RF) per model, ranked by ensemble AUROC.

| Rank | Model ID | Head | Config | Emb AUROC | Ens AUROC | AUPRC | Sensitivity | Specificity | PPV | NPV | F1 |
|------|----------|------|--------|----------:|----------:|------:|------------:|------------:|----:|----:|---:|
| 1 | `92b9afed` | LR | bbox, λ=0.1, frozen, n=all, slice | 0.614 | **0.578** | 0.817 | 0.422 | 0.625 | 0.760 | 0.278 | 0.543 |
| 2 | `5e3f71a0` | LR | raw, λ=0.1, frozen, n=all, patient | 0.534 | 0.554 | 0.784 | 0.422 | 0.625 | 0.760 | 0.278 | 0.543 |
| 3 | `a64b245f` | RF | raw, λ=0.0, frozen, n=all, slice | 0.556 | 0.542 | 0.817 | 0.422 | 0.625 | 0.760 | 0.278 | 0.543 |
| 4 | `06c598c0` | LR | raw, λ=0.0, frozen, n=all, patient | 0.515 | 0.536 | 0.794 | 0.422 | 0.625 | 0.760 | 0.278 | 0.543 |
| — | radiomic LR | LR | 149 art. features, resection-trained | — | 0.531 | 0.748 | 0.422 | 0.625 | 0.760 | 0.278 | 0.543 |
| 5 | `dc7e1d10` | RF | raw, λ=0.1, frozen, n=all, slice | 0.453 | 0.504 | 0.784 | 0.422 | 0.625 | 0.760 | 0.278 | 0.543 |

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
