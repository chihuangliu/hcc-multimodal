# Ablation Cohort Evaluation — 2-Year RFS Prediction (v5)
**Date:** 2026-08-03  
**Covers:** Soramic (ablation) cohort · Lausanne cohort  
**Based on:** `reports/0727/0727_ablation_eval_v4.md`  
**Source of models:** [`0803_full_epochs_gene_randomized.md`](0803_full_epochs_gene_randomized.md)

**Change vs 0727 v4:** the eight `dc7e1d10`-era encoders are replaced wholesale by the eight
**randomised-gene-order, 50-epoch-budget** encoders of the 0803 run set. Every model here is
evaluated at its **best-validation-loss checkpoint** (`best_model.pt`, selected over a 50-epoch
budget with `patience=2`) — not at a fixed epoch. The eight models are the full
**λ × mri_type × split-unit** 2×2×2 grid: the seven cells of the 0803 §2 grid plus **`d7085bf5`**
(raw · slice · λ=0.1), the one cell that grid did not run, taken from the 0803 §1 replicate table.
The other five §1 replicates (`09cd4b36`, `18e77da5`, `39d54fe5`, `e924f983`, `6a964bac`) are
**not** included — they are gene-order replicates of `d7085bf5`'s config, not distinct grid cells.
All tables (§2–§6) are ranked over these 8.

> **Read this first — the four patient-split runs are single-epoch encoders.** `d33f74db`,
> `26a5a902`, `5cd1cc2d` and `a2f950af` all had validation loss rise monotonically from epoch 1, so
> `patience=2` stopped them at epoch 3 and their `best_model.pt` is the **epoch-1** weights. They had
> no 50-epoch budget in any meaningful sense and are **not comparable** to the four slice-split rows,
> which trained 23–44 epochs. This is the known split-unit confound; with `patience=2` it is fatal
> rather than merely limiting. Every ranking below is contaminated by it, and the λ and mri_type
> contrasts are only clean *within* a split-unit.

---

## Table of Contents

- [1. Setup](#1-setup)
- [2. Results — Soramic](#2-results--soramic)
  - [2.1 Radiomic baselines](#21-radiomic-baselines)
  - [2.2 Embedding models — all configs](#22-embedding-models--all-configs)
  - [2.3 Summary table](#23-summary-table)
- [3. Results — Lausanne](#3-results--lausanne)
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

Unchanged from v4 — same cohorts, same labels, same radiomic feature tables.

### 1.2 Radiomic pipeline

Pre-trained on the full resection cohort (`models/radiomics/radiomic_rfs_2year_{lr,rf}.joblib`):

- 149 arterial-phase features → SelectKBest(f_classif, k=100) → classifier  
- **LR:** saga, elasticnet, l1_ratio=1.0, C=1.0  
- **RF:** max_depth=2, min_samples_leaf=10, n_estimators=100

The radiomic baselines are model-independent, so they carry over unchanged from v4.

### 1.3 Contrastive embedding pipeline

8 models evaluated on both cohorts (all frozen, n=all). Downstream head: SelectKBest(f_classif, k=100) + LR or RF fitted on resection embeddings, evaluated on test cohort embeddings. MRI: arterial phase, mean-pooled sagittal slices. Embeddings resampled to 1×1×3 mm before extraction.

#### 1.3.1 Contrastive training configurations

All 8 models share the same ViT-B/32 backbone and triplet contrastive objective, use a **frozen**
backbone, and are trained on **all** available slices per patient (**n=all**). Shared
hyperparameters: `embed_dim=128`, `gene_hidden_dim=256`, `temperature=0.07`,
`reg_mode=per_modality`, `gene_set=all` (40 genes), `n_per_axis=all` over all three axes,
`img_size=224`, `bbox_pad=10`, `val_split=0.1`, bs=32, lr=1e-4, wd=1e-4, seed=42, and a **50-epoch
budget with `patience=2`** and `checkpoint_interval=10`. **Gene column order was randomised per run**
(no `--sort_genes`); each run's resolved order is in its `metadata.json`. The axes that vary:

| Parameter | Values | Notes |
|---|---|---|
| **λ** | 0.0, 0.1 | Weight of the contrastive loss relative to the supervised classification loss. λ=0.0 trains without contrastive signal. |
| **Input** | raw, bbox | Raw full MRI vs. a bounding-box crop around the segmentation mask (`mri_type=raw_bbox`). |
| **Validation split** | slice, patient | Unit of train/val split during training. Slice-level split may allow the same patient's slices to appear in both train and val; patient-level split holds out full patients. |

The 8 model configurations (all frozen, n=all, 40-gene set, randomised gene order), with the epoch
their `best_model.pt` was written at and the number of epochs actually trained before `patience=2`
stopped the run:

| Group | Model ID | Input | Gene set | λ | Frozen | n slices | Val split | Best ep / trained |
|-------|----------|-------|----------|---|--------|----------|-----------|------------------:|
| 3 | `d7085bf5` | raw | 40 genes (rand.) | 0.1 | yes | all | slice | 42 / 44 |
| 3 | `5cd1cc2d` | raw | 40 genes (rand.) | 0.1 | yes | all | patient | **1 / 3** |
| 3 | `e837a0b4` | raw | 40 genes (rand.) | 0.0 | yes | all | slice | 36 / 38 |
| 3 | `a2f950af` | raw | 40 genes (rand.) | 0.0 | yes | all | patient | **1 / 3** |
| 4 | `78456720` | bbox | 40 genes (rand.) | 0.1 | yes | all | slice | 30 / 32 |
| 4 | `d33f74db` | bbox | 40 genes (rand.) | 0.1 | yes | all | patient | **1 / 3** |
| 4 | `41c6db8a` | bbox | 40 genes (rand.) | 0.0 | yes | all | slice | 21 / 23 |
| 4 | `26a5a902` | bbox | 40 genes (rand.) | 0.0 | yes | all | patient | **1 / 3** |

Group 3: raw MRI input; Group 4: bounding-box crop input. Both groups span the full
λ ∈ {0.0, 0.1} × split ∈ {slice, patient} grid, so the 8 rows are the complete 2×2×2 design.
Bold "1 / 3" marks the four epoch-1 encoders flagged at the top of this report.

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

#### Group 3 — Raw MRI input (n=all, frozen backbone)

| Config | Model ID | LR AUROC | RF AUROC | Best AUROC |
|--------|----------|--------:|--------:|----------:|
| λ=0.1, frozen, n=all, slice | `d7085bf5` | 0.695 | 0.457 | 0.695 |
| λ=0.1, frozen, n=all, patient | `5cd1cc2d` | 0.701 | 0.650 | **0.701** |
| λ=0.0, frozen, n=all, slice | `e837a0b4` | 0.553 | 0.506 | 0.553 |
| λ=0.0, frozen, n=all, patient | `a2f950af` | 0.637 | 0.457 | 0.637 |

#### Group 4 — Bounding box

| Config | Model ID | LR AUROC | RF AUROC | Best AUROC |
|--------|----------|--------:|--------:|----------:|
| λ=0.1, frozen, n=all, slice | `78456720` | 0.326 | 0.316 | 0.326 |
| λ=0.1, frozen, n=all, patient | `d33f74db` | 0.477 | 0.616 | 0.616 |
| λ=0.0, frozen, n=all, slice | `41c6db8a` | 0.477 | 0.434 | 0.477 |
| λ=0.0, frozen, n=all, patient | `26a5a902` | 0.557 | 0.655 | 0.655 |

### 2.3 Summary table

Best head (LR or RF, whichever has higher AUROC) per model, ranked by AUROC. Multi-lesion: average. Threshold: 0.5. "—" = undefined NPV (all samples predicted positive).

| Rank | Model ID | Head | Config | AUROC | AUPRC | Sensitivity | Specificity | PPV | NPV | F1 |
|------|----------|------|--------|------:|------:|------------:|------------:|----:|----:|---:|
| 1 | `5cd1cc2d` | LR | raw, λ=0.1, frozen, n=all, patient | **0.701** | 0.846 | 0.385 | 0.833 | 0.833 | 0.385 | 0.526 |
| 2 | `d7085bf5` | LR | raw, λ=0.1, frozen, n=all, slice | 0.695 | 0.812 | 0.923 | 0.389 | 0.766 | 0.700 | 0.837 |
| 3 | `26a5a902` | RF | bbox, λ=0.0, frozen, n=all, patient | 0.655 | 0.802 | 0.868 | 0.111 | 0.673 | 0.286 | 0.759 |
| 4 | `a2f950af` | LR | raw, λ=0.0, frozen, n=all, patient | 0.637 | 0.809 | 0.744 | 0.500 | 0.763 | 0.474 | 0.753 |
| 5 | `d33f74db` | RF | bbox, λ=0.1, frozen, n=all, patient | 0.616 | 0.768 | 1.000 | 0.000 | 0.679 | — | 0.809 |
| — | radiomic RF | RF | 149 art. features, resection-trained | 0.590 | 0.766 | 0.143 | 1.000 | 1.000 | 0.375 | 0.250 |
| 6 | `e837a0b4` | LR | raw, λ=0.0, frozen, n=all, slice | 0.553 | 0.738 | 0.718 | 0.389 | 0.718 | 0.389 | 0.718 |
| — | radiomic LR | LR | 149 art. features, resection-trained | 0.518 | 0.671 | 0.657 | 0.389 | 0.676 | 0.368 | 0.667 |
| 7 | `41c6db8a` | LR | bbox, λ=0.0, frozen, n=all, slice | 0.477 | 0.683 | 0.868 | 0.000 | 0.647 | 0.000 | 0.742 |
| 8 | `78456720` | LR | bbox, λ=0.1, frozen, n=all, slice | 0.326 | 0.624 | 0.974 | 0.000 | 0.673 | 0.000 | 0.796 |

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

#### Group 3 — Raw MRI input (n=all, frozen backbone)

| Config | Model ID | LR AUROC | RF AUROC | Best AUROC |
|--------|----------|--------:|--------:|----------:|
| λ=0.1, frozen, n=all, slice | `d7085bf5` | 0.429 | 0.498 | 0.498 |
| λ=0.1, frozen, n=all, patient | `5cd1cc2d` | 0.492 | 0.441 | 0.492 |
| λ=0.0, frozen, n=all, slice | `e837a0b4` | 0.433 | 0.526 | 0.526 |
| λ=0.0, frozen, n=all, patient | `a2f950af` | 0.510 | 0.497 | 0.510 |

#### Group 4 — Bounding box

| Config | Model ID | LR AUROC | RF AUROC | Best AUROC |
|--------|----------|--------:|--------:|----------:|
| λ=0.1, frozen, n=all, slice | `78456720` | 0.453 | 0.388 | 0.453 |
| λ=0.1, frozen, n=all, patient | `d33f74db` | 0.406 | 0.537 | 0.537 |
| λ=0.0, frozen, n=all, slice | `41c6db8a` | 0.602 | 0.580 | **0.602** |
| λ=0.0, frozen, n=all, patient | `26a5a902` | 0.495 | 0.558 | 0.558 |

### 3.3 Summary table

Best head (LR or RF, whichever has higher AUROC) per model, ranked by AUROC. Multi-lesion: average. Threshold: 0.5. "—" = undefined NPV (all samples predicted positive).

| Rank | Model ID | Head | Config | AUROC | AUPRC | Sensitivity | Specificity | PPV | NPV | F1 |
|------|----------|------|--------|------:|------:|------------:|------------:|----:|----:|---:|
| 1 | `41c6db8a` | LR | bbox, λ=0.0, frozen, n=all, slice | **0.602** | 0.836 | 0.896 | 0.235 | 0.768 | 0.444 | 0.827 |
| 2 | `26a5a902` | RF | bbox, λ=0.0, frozen, n=all, patient | 0.558 | 0.798 | 0.979 | 0.000 | 0.734 | 0.000 | 0.839 |
| 3 | `d33f74db` | RF | bbox, λ=0.1, frozen, n=all, patient | 0.537 | 0.768 | 1.000 | 0.000 | 0.738 | — | 0.850 |
| — | radiomic LR | LR | 149 art. features, resection-trained | 0.531 | 0.748 | 0.422 | 0.625 | 0.760 | 0.278 | 0.543 |
| 4 | `e837a0b4` | RF | raw, λ=0.0, frozen, n=all, slice | 0.526 | 0.746 | 1.000 | 0.000 | 0.742 | — | 0.852 |
| 5 | `a2f950af` | LR | raw, λ=0.0, frozen, n=all, patient | 0.510 | 0.750 | 0.551 | 0.588 | 0.794 | 0.312 | 0.651 |
| — | radiomic RF | RF | 149 art. features, resection-trained | 0.506 | 0.747 | 0.067 | 0.875 | 0.600 | 0.250 | 0.120 |
| 6 | `d7085bf5` | RF | raw, λ=0.1, frozen, n=all, slice | 0.498 | 0.771 | 1.000 | 0.000 | 0.742 | — | 0.852 |
| 7 | `5cd1cc2d` | LR | raw, λ=0.1, frozen, n=all, patient | 0.492 | 0.752 | 0.347 | 0.706 | 0.773 | 0.273 | 0.479 |
| 8 | `78456720` | LR | bbox, λ=0.1, frozen, n=all, slice | 0.453 | 0.744 | 0.958 | 0.000 | 0.730 | 0.000 | 0.829 |

---

## 4. Rank by CV AUC

In-CV AUC uses **fixed LR/RF, no hyperparameter search, no feature selection**, all on the **same
image extraction as the transfer columns** — the survival `resection_img_emb.parquet` cache (128-dim
image-only, patient-level mean-pooled, written from each run's `best_model.pt`). LR/RF are taken
as-is from `hcc_multimodal/baselines/config.py` (`MODELS`; LR = saga, elasticnet, l1_ratio=1.0,
C=1.0, max_iter=5000; RF = 100 trees, default depth) and fit on **all 128 image dims** through
`SimpleImputer(median) → StandardScaler → classifier` in a plain **3-fold** stratified CV on the
54-patient (26 positive) resection cohort. "Best head" = higher mean fold AUC. Soramic/Lausanne
columns are the **CV-selected head's own** transfer — that same head refit on all resection and
applied to each cohort through the identical all-128 no-FS pipeline (not the §2–3
SelectKBest(k=100), best-of-{LR,RF} numbers, which mix a different head per cohort). Δ vs. best
radiomic baseline (Soramic RF=0.590; Lausanne LR=0.531).

| CV Rank | Model ID | Config | Best head | CV AUC ± std | Soramic AUROC | Lausanne AUROC |
|--------:|----------|--------|-----------|-------------:|--------------:|---------------:|
| 1 | `d7085bf5` | raw, λ=0.1, frozen, n=all, slice | LR | **0.694 ± 0.040** | **0.726 (+0.136)** | 0.424 (−0.107) |
| 2 | `78456720` | bbox, λ=0.1, frozen, n=all, slice | LR | 0.636 ± 0.119 | 0.349 (−0.241) | 0.456 (−0.075) |
| 3 | `5cd1cc2d` | raw, λ=0.1, frozen, n=all, patient | LR | 0.628 ± 0.063 | 0.678 (+0.088) | 0.483 (−0.048) |
| 4 | `26a5a902` | bbox, λ=0.0, frozen, n=all, patient | RF | 0.610 ± 0.034 | 0.488 (−0.102) | 0.481 (−0.050) |
| 5 | `41c6db8a` | bbox, λ=0.0, frozen, n=all, slice | RF | 0.579 ± 0.026 | 0.421 (−0.169) | 0.491 (−0.040) |
| 6 | `a2f950af` | raw, λ=0.0, frozen, n=all, patient | LR | 0.578 ± 0.097 | 0.638 (+0.048) | **0.510 (−0.021)** |
| 7 | `d33f74db` | bbox, λ=0.1, frozen, n=all, patient | RF | 0.529 ± 0.069 | 0.597 (+0.007) | 0.503 (−0.028) |
| 8 | `e837a0b4` | raw, λ=0.0, frozen, n=all, slice | LR | 0.521 ± 0.100 | 0.554 (−0.036) | 0.433 (−0.098) |

**`d7085bf5` is CV rank 1 by a clear margin (0.694 vs 0.636 for the runner-up) and is the only
model of the eight whose transfer justifies it on Soramic (0.726, +0.136 over radiomic RF).** Its CV
spread is also the tightest of the top three (±0.040 vs ±0.119 for `78456720`), so the lead is not a
fold artefact. It remains weak on Lausanne (0.424, −0.107), reprising the cross-cohort split seen in
every prior version of this report.

**CV rank and Soramic transfer disagree everywhere below rank 1.** `78456720` (CV rank 2, 0.636)
transfers at **0.349** on Soramic — the worst of the eight and far below chance — while `a2f950af`
(CV rank 6) and `d33f74db` (CV rank 7) both transfer *above* the radiomic RF baseline. Resection CV
on 54 patients is not a reliable selector for this cohort at these CV separations (the seven
non-leading models sit within 0.12 of each other, 0.521–0.636).

**No model beats the radiomic Lausanne baseline on the all-128 pipeline.** Every Lausanne Δ is
negative; the closest is `a2f950af` at −0.021. This is stronger than the v4 picture, where
`92b9afed` cleared the baseline by +0.060 — the randomised-gene-order/best-checkpoint family does
not reproduce that.

**λ and mri_type give no consistent signal once split-unit is held fixed.** Among the four
slice-split runs (the only ones with a real training budget), CV goes raw·λ=0.1 0.694 >
bbox·λ=0.1 0.636 > bbox·λ=0.0 0.579 > raw·λ=0.0 0.521 — λ=0.1 beats λ=0.0 in both input types, but
the input-type ordering flips with λ. Among the four patient-split (epoch-1) runs the ordering is
raw·λ=0.1 0.628 > bbox·λ=0.0 0.610 > raw·λ=0.0 0.578 > bbox·λ=0.1 0.529, i.e. the λ effect reverses
for bbox. With one run per cell and no seed replicates these differences are not separable from
noise.

---

## 5. Observations

### 5.1 Soramic

1. **The two λ=0.1 raw configs are the top two (0.695–0.701)**: `5cd1cc2d` (LR, 0.701, patient
   split) and `d7085bf5` (LR, 0.695, slice split). They are the only two models clearly above the
   0.65 band, and both pair raw MRI input with the contrastive term.

2. **All four raw configs beat their bbox counterparts at matched λ and split, except the λ=0.0
   patient cell.** raw 0.695 vs bbox 0.326 (λ=0.1, slice), raw 0.701 vs bbox 0.616 (λ=0.1, patient),
   raw 0.553 vs bbox 0.477 (λ=0.0, slice); only at λ=0.0/patient does bbox win (0.655 vs 0.637). The
   bbox crop is a net loss on Soramic in this family — the opposite of what the v4 bbox models showed
   on Lausanne.

3. **`78456720` (bbox, λ=0.1, slice) collapses to 0.326** — well below chance and the worst
   embedding result in any version of this report. Its scores are not merely uninformative but
   *anti-correlated* with outcome on Soramic, despite a middling resection CV of 0.636 (rank 2).

4. **Radiomic RF (0.590) sits between ranks 5 and 6**, above `e837a0b4`, `41c6db8a` and `78456720`
   and below the five models at 0.616–0.701. Three of eight embedding models fail to clear it.

5. **The patient-split (epoch-1) encoders are not obviously worse on Soramic** — they occupy ranks
   1, 3, 4 and 5. Read with the caveat at the top of this report: an epoch-1 frozen-ViT encoder is
   close to a lightly-perturbed pre-trained feature extractor, and on this cohort that is competitive
   with a 42-epoch fine-tune. It is evidence about the *contrastive training*, not about the
   split-unit axis being harmless.

6. **Head choice is mixed and tracks the split unit.** LR wins for all four raw configs and
   `41c6db8a`/`78456720`; RF wins only for the two bbox patient-split models (`d33f74db`,
   `26a5a902`).

### 5.2 Lausanne

1. **Overall performance is at or below chance for most models, and the top three are all bbox.**
   `41c6db8a` (LR, 0.602) leads, followed by `26a5a902` (RF, 0.558) and `d33f74db` (RF, 0.537).
   Seven of eight embedding models fall below 0.56.

2. **The Soramic ordering inverts almost exactly.** `78456720` is last on both cohorts, but the two
   Soramic leaders `5cd1cc2d` (0.701 → 0.492, rank 7) and `d7085bf5` (0.695 → 0.498, rank 6) drop to
   the bottom third of Lausanne, while the Soramic rank-7 `41c6db8a` (0.477) leads Lausanne (0.602).
   The two cohorts continue to pull in opposite directions, as in v4.

3. **The bbox crop helps Lausanne — but here only at λ=0.0.** The two λ=0.0 bbox models take
   Lausanne ranks 1 and 2 (0.558–0.602); the two λ=0.1 bbox models are 0.453 and 0.537. This is the
   **reverse** of v4, where the bbox advantage on Lausanne was contingent on λ=0.1. Across the two
   report versions the bbox × λ interaction has no stable sign.

4. **Only one embedding model beats the radiomic LR baseline (0.531): `41c6db8a` (0.602, +0.071).**
   `26a5a902` (0.558) is the only other above it. The remaining six trail both radiomic baselines or
   sit between them.

5. **Both radiomic baselines are again near chance (0.506–0.531)** and now outrank five of the eight
   embedding models. Neither pipeline generalises to the Lausanne `MRI_liver_arterial` acquisition.

### 5.3 Cross-cohort

**No configuration in the full 2×2×2 grid transfers to both cohorts.** The Soramic-Lausanne rank
correlation over the eight models is strongly negative: the five models above 0.60 on Soramic
average 0.519 on Lausanne — below the radiomic LR baseline — and the Lausanne leader is Soramic
rank 7. Picking on resection CV gives
`d7085bf5`, which is the best available Soramic model (+0.136 over baseline) and a below-baseline
Lausanne model (−0.033 vs radiomic LR). That is the same trade v4 reported for `dc7e1d10`; the
randomised-gene-order, best-checkpoint retraining did not resolve it.

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
| 1 | `d7085bf5` | LR | raw, λ=0.1, frozen, n=all, slice | 0.695 | **0.725** | 0.829 | 0.886 | 0.389 | 0.738 | 0.636 | 0.805 |
| 2 | `5cd1cc2d` | LR | raw, λ=0.1, frozen, n=all, patient | 0.701 | 0.719 | 0.844 | 0.371 | 0.889 | 0.867 | 0.421 | 0.520 |
| 3 | `a2f950af` | LR | raw, λ=0.0, frozen, n=all, patient | 0.637 | 0.641 | 0.797 | 0.714 | 0.500 | 0.735 | 0.474 | 0.725 |
| 4 | `26a5a902` | RF | bbox, λ=0.0, frozen, n=all, patient | 0.655 | 0.640 | 0.780 | 0.829 | 0.333 | 0.707 | 0.500 | 0.763 |
| 5 | `d33f74db` | RF | bbox, λ=0.1, frozen, n=all, patient | 0.616 | 0.627 | 0.776 | 1.000 | 0.000 | 0.660 | — | 0.795 |
| — | radiomic RF | RF | 149 art. features, resection-trained | — | 0.590 | 0.766 | 0.143 | 1.000 | 1.000 | 0.375 | 0.250 |
| 6 | `e837a0b4` | RF | raw, λ=0.0, frozen, n=all, slice | 0.553 | 0.544 | 0.745 | 1.000 | 0.000 | 0.660 | — | 0.795 |
| 7 | `41c6db8a` | LR | bbox, λ=0.0, frozen, n=all, slice | 0.477 | 0.508 | 0.711 | 0.829 | 0.000 | 0.617 | 0.000 | 0.707 |
| 8 | `78456720` | LR | bbox, λ=0.1, frozen, n=all, slice | 0.326 | 0.492 | 0.694 | 0.943 | 0.000 | 0.647 | 0.000 | 0.767 |

The ensemble is a **floor-raiser, not a ceiling-raiser**: it lifts the four models that were below
the radiomic baseline (`78456720` 0.326 → 0.492, `41c6db8a` 0.477 → 0.508) while leaving the two
leaders essentially where they were (`d7085bf5` 0.695 → 0.725, `5cd1cc2d` 0.701 → 0.719). Only
`d7085bf5` gains materially (+0.030), and it takes rank 1 from `5cd1cc2d`.

### 6.2 Lausanne — ensemble summary

Best ensemble head (LR or RF) per model, ranked by ensemble AUROC.

| Rank | Model ID | Head | Config | Emb AUROC | Ens AUROC | AUPRC | Sensitivity | Specificity | PPV | NPV | F1 |
|------|----------|------|--------|----------:|----------:|------:|------------:|------------:|----:|----:|---:|
| 1 | `41c6db8a` | LR | bbox, λ=0.0, frozen, n=all, slice | 0.602 | **0.575** | 0.804 | 0.422 | 0.625 | 0.760 | 0.278 | 0.543 |
| 2 | `d33f74db` | RF | bbox, λ=0.1, frozen, n=all, patient | 0.537 | 0.563 | 0.746 | 0.422 | 0.625 | 0.760 | 0.278 | 0.543 |
| 3 | `26a5a902` | RF | bbox, λ=0.0, frozen, n=all, patient | 0.558 | 0.540 | 0.783 | 0.422 | 0.625 | 0.760 | 0.278 | 0.543 |
| 4 | `e837a0b4` | RF | raw, λ=0.0, frozen, n=all, slice | 0.526 | 0.534 | 0.794 | 0.422 | 0.625 | 0.760 | 0.278 | 0.543 |
| — | radiomic LR | LR | 149 art. features, resection-trained | — | 0.531 | 0.748 | 0.422 | 0.625 | 0.760 | 0.278 | 0.543 |
| 5 | `d7085bf5` | RF | raw, λ=0.1, frozen, n=all, slice | 0.498 | 0.526 | 0.791 | 0.422 | 0.625 | 0.760 | 0.278 | 0.543 |
| 6 | `5cd1cc2d` | LR | raw, λ=0.1, frozen, n=all, patient | 0.492 | 0.524 | 0.788 | 0.422 | 0.625 | 0.760 | 0.278 | 0.543 |
| 7 | `a2f950af` | LR | raw, λ=0.0, frozen, n=all, patient | 0.510 | 0.519 | 0.772 | 0.422 | 0.625 | 0.760 | 0.278 | 0.543 |
| 8 | `78456720` | LR | bbox, λ=0.1, frozen, n=all, slice | 0.453 | 0.512 | 0.763 | 0.422 | 0.625 | 0.760 | 0.278 | 0.543 |

As in v4, **every Lausanne ensemble row has identical threshold metrics** (sens 0.422 / spec 0.625 /
F1 0.543 — the radiomic LR's own operating point): at threshold 0.5 the averaged probability is
dominated by the radiomic component, so only AUROC distinguishes the rows. Ensembling compresses the
whole field into 0.512–0.575, pulling the weak models up and the leader down (`41c6db8a`
0.602 → 0.575). Only three models finish above the radiomic LR baseline, and none by more than
0.044.

---

## 7. File references

| Artifact | Path |
|---|---|
| Radiomic LR/RF (Soramic) | `results/eval/soramic/radiomic_{lr,rf}_rfs_2year_20260530_020430.json` |
| Embedding results — Soramic | `results/eval/soramic/embedding_{model_id}_rfs_2year_{timestamp}.json` |
| Ensemble results — Soramic | `results/eval/soramic/ensemble_{model_id}_rfs_2year_bestckpt.json` |
| Radiomic LR/RF (Lausanne) | `results/eval/lusanne/radiomic_rfs_2year_{timestamp}.json` |
| Embedding results — Lausanne | `results/eval/lusanne/embedding_{model_id}_rfs_2year_{timestamp}.json` |
| Ensemble results — Lausanne | `results/eval/lusanne/ensemble_{model_id}_rfs_2year_bestckpt.json` |
| §4 CV-rank CSVs | `results/eval/cv_rank_0803_bestckpt/cv_rank_image_only.csv` (`d7085bf5`), `results/eval/cv_rank_lamgrid/cv_rank_image_only.csv` (other 7) |
| Radiomic models | `models/radiomics/radiomic_rfs_2year_{lr,rf}.joblib` |
| Contrastive models | `training/contrastive/{model_id}/best_model.pt` (+ `losses.csv`, `metadata.json` incl. `gene_order`) |
| Cached resection embeddings | `training/contrastive/{model_id}/cached_embeddings/resection_img_emb.parquet` |
| Cached Soramic embeddings | `training/contrastive/{model_id}/cached_embeddings/ablation_soramic_img_emb_{raw,bbox}.parquet` |
| Cached Lausanne embeddings | `training/contrastive/{model_id}/cached_embeddings/ablation_lusanne_img_emb_{raw,bbox}.parquet` |
| Training runners | `scripts/run_lam_mri_split_grid_local.sh` (7 grid cells), `scripts/eval_contrastive_runs.py` |
| Companion grid report | [`0803_embedding_grid_eval_v5.md`](0803_embedding_grid_eval_v5.md) |

> **Cache/checkpoint note.** For every model in this report the *unsuffixed* cache files above are
> the `best_model.pt` extraction. `d7085bf5` additionally has `…__ep010.parquet` caches (its
> epoch-10 checkpoint, used in the 0803 §1 table); those are **not** used here.

Regenerate — §2/§3 embedding evals (one invocation per model × cohort; `--overwrite-cache` only if
the embedding caches need re-extraction):
```
for m in d7085bf5 78456720 41c6db8a d33f74db 26a5a902 5cd1cc2d a2f950af e837a0b4; do
  for c in soramic lusanne; do
    python -m hcc_multimodal.eval.eval --mode embedding --model-id $m --ablation-set $c \
      --target rfs_2year --multi-lesion both --select-k 100
  done
done
```
Regenerate — §6 ensembles (radiomic component fixed per cohort):
```
for m in d7085bf5 78456720 41c6db8a d33f74db 26a5a902 5cd1cc2d a2f950af e837a0b4; do
  python -m hcc_multimodal.eval.eval --mode ensemble --model-id $m --ablation-set soramic \
    --radiomic-model models/radiomics/radiomic_rfs_2year_rf.joblib --target rfs_2year \
    --multi-lesion average --output results/eval/soramic/ensemble_${m}_rfs_2year_bestckpt.json
  python -m hcc_multimodal.eval.eval --mode ensemble --model-id $m --ablation-set lusanne \
    --radiomic-model models/radiomics/radiomic_rfs_2year_lr.joblib --target rfs_2year \
    --multi-lesion average --output results/eval/lusanne/ensemble_${m}_rfs_2year_bestckpt.json
done
```
Regenerate — §4 CV rank:
```
python -m hcc_multimodal.eval.embedding_grid_eval --task cv-rank \
  --model-ids d7085bf5 78456720 41c6db8a d33f74db 26a5a902 5cd1cc2d a2f950af e837a0b4 \
  --output-dir results/eval/cv_rank_0803_v5
```
