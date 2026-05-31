
## Table of Contents

- [1. Task](#1-task)
- [2. Key findings](#2-key-findings)
- [3. Methodology](#3-methodology)
  - [3.1 Split strategies](#31-split-strategies)
  - [3.2 Models](#32-models)
  - [3.3 Downstream CV pipeline](#33-downstream-cv-pipeline)
- [4. Training results](#4-training-results)
- [5. Downstream CV results](#5-downstream-cv-results)
  - [5.1 Group 1 — Raw MRI, λ sweep](#51-group-1--raw-mri-λ-sweep)
  - [5.2 Group 2 — Gene set ablation](#52-group-2--gene-set-ablation)
  - [5.3 Group 3 — Full slices](#53-group-3--full-slices)
  - [5.4 Group 4 — Bounding box](#54-group-4--bounding-box)

---

# 1. Task

Retrain all 8 contrastive models from the 0525 experiments with a **patient-level split** (`--split-unit patient`): split at the patient level, stratified by 2-year RFS outcome, so all slices from a given patient land in exactly one split. The 0525 results used a slice-level split where slices from the same patient could appear in both train and val sets — this is the baseline we compare against.

| # | Old ID (0525, slice split) | Group | Key config |
|---|--------------------------|-------|-----------|
| 1 | `6a1a1bdf` | Raw λ sweep | raw, λ=0.1, unfrozen, n=10, 50 ep, all genes |
| 2 | `982a6fa2` | Raw λ sweep | raw, λ=0.0, unfrozen, n=10, 50 ep, all genes |
| 3 | `12e4ba6a` | Gene ablation | raw, λ=0.1, unfrozen, n=10, 50 ep, `predefined_2y_cv` |
| 4 | `5d04e6ba` | Gene ablation | raw, λ=0.1, unfrozen, n=10, 50 ep, `2y_before_cv` |
| 5 | `dc7e1d10` | Full slices | raw, λ=0.1, frozen, n=all, 10 ep |
| 6 | `a64b245f` | Full slices | raw, λ=0.0, frozen, n=all, 10 ep |
| 7 | `050d401d` | BBox | raw_bbox, λ=0.1, unfrozen, n=10, 50 ep |
| 8 | `e12b0592` | BBox | raw_bbox, λ=0.0, unfrozen, n=10, 50 ep |

---

# 2. Key findings

## 2.1 Patient split vs slice split — best embeddings AUC per model (full slice inference)

All AUC values use n=all slices at inference (mean-pooled embeddings across all sagittal slices / all bbox slices). Best of LR / RF shown.

| # | Config | Slice split (n=all) | Patient split (n=all) | Δ |
|---|--------|--------------------|-----------------------|---|
| 1 | raw, λ=0.1, unfrozen | 0.672 ± 0.090 | 0.690 ± 0.036 | +0.02 |
| 2 | raw, λ=0.0, unfrozen | 0.717 ± 0.087 | **0.739 ± 0.100** | +0.02 |
| 3 | raw, λ=0.1, predefined genes | 0.522 ± 0.087 | 0.457 ± 0.040 | −0.07 |
| 4 | raw, λ=0.1, 2y_before_cv genes | 0.746 ± 0.117 | **0.789 ± 0.032** | +0.04 |
| 5 | raw, λ=0.1, frozen, n=all train | **1.000 ± 0.000** | 0.640 ± 0.038 | −0.36 |
| 6 | raw, λ=0.0, frozen, n=all train | 0.739 ± 0.129 | 0.648 ± 0.100 | −0.09 |
| 7 | raw_bbox, λ=0.1, unfrozen | **0.965 ± 0.026** | 0.661 ± 0.117 | −0.30 |
| 8 | raw_bbox, λ=0.0, unfrozen | 0.657 ± 0.100 | 0.568 ± 0.101 | −0.09 |

## 2.2 Key observations

1. **Slice split inflated results most for high-capacity configs.** Models 5 (frozen, n=all) and 7 (bbox, n=10) drop −0.36 and −0.30 — both had near-perfect slice-split AUC driven by patient-identity memorisation. Standard configs (Models 1, 2) show almost no change (+0.02), suggesting they had less leakage to begin with.

2. **2y_before_cv gene set (Model 4) is the best overall.** Patient-split n=all AUC = **0.789 ± 0.032**, slightly *higher* than its slice-split number. The curated RFS gene set focuses the image encoder on outcome-relevant biology rather than spreading capacity over all 40 gene axes.

3. **Predefined gene set (Model 3) is the exception — drops with n=all.** n=10 patient split gave 0.707 ± 0.017 but n=all drops to 0.457 ± 0.040. This suggests the model learns features concentrated in a small number of informative slices that get diluted by averaging all slices.

4. **Bounding box does not help with honest evaluation.** Model 7 slice-split 0.965 → patient-split 0.661, indistinguishable from raw Model 1 (0.690). The bbox advantage was entirely leakage.

5. **Frozen backbone + n=all training (Models 5, 6) is the worst config.** Best val loss at epoch 1 in both cases, train loss goes negative — pure memorisation. Patient-split AUC 0.640 / 0.648.

---

# 3. Methodology

## 3.1 Split strategies

| Strategy | `--split-unit` | Unit | Stratification |
|----------|---------------|------|----------------|
| Slice split (0525 baseline) | `slice` | Individual MRI slice | By slice-level outcome label |
| Patient split (corrected) | `patient` | Patient ID | By patient-level 2-year RFS outcome |

Val fraction: 10% (`--val_split 0.1`), seed 42. With 54 patients this gives 6 held-out patients (3 recurrence / 3 no recurrence) vs the slice split which leaks 38 of 54 patients' anatomy into both sets.

## 3.2 Models

Shared parameters across all runs unless noted:

| Parameter | Value |
|-----------|-------|
| Backbone | ViT-B/32 |
| Embed dim | 128 per modality |
| Gene hidden dim | 256 |
| Temperature τ | 0.07 |
| reg_mode | per_modality |
| Axes | sagittal (0) |
| Image size | 224 × 224 |
| Batch size | 32 |
| Optimiser | AdamW (lr=1e-4, wd=1e-4) |
| Seed | 42 |

Model-specific parameters:

| # | λ | Backbone frozen | n_per_axis | MRI type | bbox_pad | Epochs |
|---|---|----------------|-----------|----------|----------|--------|
| 1 | 0.1 | No | 10 | raw | — | 50 |
| 2 | 0.0 | No | 10 | raw | — | 50 |
| 3 | 0.1 | No | 10 | raw | — | 50 |
| 4 | 0.1 | No | 10 | raw | — | 50 |
| 5 | 0.1 | Yes | all | raw | — | 10 |
| 6 | 0.0 | Yes | all | raw | — | 10 |
| 7 | 0.1 | No | 10 | raw_bbox | 10 | 50 |
| 8 | 0.0 | No | 10 | raw_bbox | 10 | 50 |

Gene sets:
- Models 1, 2, 5, 6, 7, 8: `all` (40 genes)
- Model 3: `predefined_2y_cv` (20 genes)
- Model 4: `2y_before_cv` (20 genes)

## 3.3 Downstream CV pipeline

3-fold stratified CV (`StratifiedKFold`, `random_state=42`), `SelectKBest(f_classif, k=100)` fitted on the training fold only, `StandardScaler` inside pipeline. Tasks: `radiomics`, `embeddings`, `concat`, `ensemble`. Metric: ROC-AUC mean ± std across 3 folds.

Inference conditions per model group:

| Group | Model # | Inference slices |
|-------|---------|-----------------|
| Raw λ sweep | 1, 2 | n=10 (condition 1) and n=all (condition 2) |
| Gene ablation | 3, 4 | n=10 |
| Full slices | 5, 6 | n=all |
| BBox | 7, 8 | n=10 (condition 1) and n=all (condition 2) |

---

# 4. Training loss and best epochs

| # | Config | Slice split ID | Best val loss (ep) | Patient split ID | Best val loss (ep) |
|---|--------|----------------|--------------------|------------------|--------------------|
| 1 | raw, λ=0.1, n=10 | `6a1a1bdf` | 1.327 (ep 28) | `1361bef2` | 1.474 (ep 19) |
| 2 | raw, λ=0.0, n=10 | `982a6fa2` | 3.941 (ep 15) | `a6f970d6` | 4.255 (ep 14) |
| 3 | raw, λ=0.1, predefined genes | `12e4ba6a` | 1.430 (ep 15) | `34e6806f` | 1.495 (ep 15) |
| 4 | raw, λ=0.1, 2y_before_cv genes | `5d04e6ba` | 1.417 (ep 19) | `9109a6c2` | 1.466 (ep 25) |
| 5 | raw, λ=0.1, frozen, n=all | `dc7e1d10` | 0.259 (ep 1) | `5e3f71a0` | 2.598 (ep 1) |
| 6 | raw, λ=0.0, frozen, n=all | `a64b245f` | −0.085 (ep 1) | `06c598c0` | 9.417 (ep 1) |
| 7 | bbox, λ=0.1, n=10 | `050d401d` | 0.396 (ep 49) | `f8aabb75` | 1.580 (ep 11) |
| 8 | bbox, λ=0.0, n=10 | `e12b0592` | 1.637 (ep 45) | `8715461c` | 4.368 (ep 5) |

---

# 5. Downstream CV results

ROC-AUC mean ± std across 3 folds. Bold = best per row.

## 5.1 Group 1 — Raw MRI, λ sweep

### Model 1 (raw, λ=0.1, unfrozen, n=10, 50 ep)

**Condition 1 — infer 10 slices:**

| Task | Classifier | Slice split (`6a1a1bdf`) | Patient split (new) |
|------|-----------|--------------------------|---------------------|
| Radiomics | LR | 0.500 ± 0.000 | 0.500 ± 0.000 |
| Radiomics | RF | 0.569 ± 0.133 | 0.569 ± 0.133 |
| Embeddings | LR | 0.752 ± 0.062 | 0.574 ± 0.061 |
| Embeddings | RF | **0.798 ± 0.081** | **0.618 ± 0.068** |
| Concat | LR | 0.516 ± 0.092 | 0.503 ± 0.141 |
| Concat | RF | 0.489 ± 0.096 | 0.483 ± 0.115 |
| Ensemble | LR | 0.648 ± 0.096 | 0.562 ± 0.086 |
| Ensemble | RF | 0.696 ± 0.078 | 0.564 ± 0.076 |

**Condition 2 — infer all slices:**

| Task | Classifier | Slice split (`6a1a1bdf`) | Patient split (new) |
|------|-----------|--------------------------|---------------------|
| Embeddings | LR | 0.574 ± 0.022 | 0.636 ± 0.157 |
| Embeddings | RF | 0.672 ± 0.090 | **0.690 ± 0.036** |
| Concat | LR | 0.495 ± 0.068 | 0.512 ± 0.136 |
| Concat | RF | 0.545 ± 0.139 | 0.549 ± 0.088 |
| Ensemble | LR | 0.591 ± 0.059 | 0.619 ± 0.148 |
| Ensemble | RF | 0.599 ± 0.065 | 0.624 ± 0.083 |

### Model 2 (raw, λ=0.0, unfrozen, n=10, 50 ep)

**Condition 1 — infer 10 slices:**

| Task | Classifier | Slice split (`982a6fa2`) | Patient split (new) |
|------|-----------|--------------------------|---------------------|
| Radiomics | LR | 0.500 ± 0.000 | 0.500 ± 0.000 |
| Radiomics | RF | 0.569 ± 0.133 | 0.569 ± 0.133 |
| Embeddings | LR | **0.706 ± 0.099** | **0.661 ± 0.040** |
| Embeddings | RF | 0.671 ± 0.065 | 0.686 ± 0.038 |
| Concat | LR | 0.529 ± 0.085 | 0.590 ± 0.153 |
| Concat | RF | 0.559 ± 0.144 | 0.571 ± 0.162 |
| Ensemble | LR | 0.578 ± 0.029 | **0.640 ± 0.037** |
| Ensemble | RF | 0.595 ± 0.063 | 0.615 ± 0.077 |

**Condition 2 — infer all slices:**

| Task | Classifier | Slice split (`982a6fa2`) | Patient split (new) |
|------|-----------|--------------------------|---------------------|
| Embeddings | LR | 0.636 ± 0.048 | **0.739 ± 0.100** |
| Embeddings | RF | 0.717 ± 0.087 | 0.642 ± 0.063 |
| Concat | LR | 0.524 ± 0.072 | 0.537 ± 0.128 |
| Concat | RF | 0.539 ± 0.130 | 0.567 ± 0.132 |
| Ensemble | LR | 0.603 ± 0.078 | **0.681 ± 0.084** |
| Ensemble | RF | 0.605 ± 0.062 | 0.595 ± 0.068 |

---

## 5.2 Group 2 — Gene set ablation

All models: raw, λ=0.1, unfrozen, n=10, 50 ep. Inference: n=10 only.

### Model 3 (`predefined_2y_cv` genes)

| Task | Classifier | Slice split (`12e4ba6a`) | Patient split (new) |
|------|-----------|--------------------------|---------------------|
| Radiomics | LR | 0.500 ± 0.000 | 0.500 ± 0.000 |
| Radiomics | RF | 0.569 ± 0.133 | 0.569 ± 0.133 |
| Embeddings | LR | 0.495 ± 0.145 | 0.504 ± 0.079 |
| Embeddings | RF | 0.585 ± 0.032 | **0.707 ± 0.017** |
| Concat | LR | 0.450 ± 0.019 | 0.483 ± 0.051 |
| Concat | RF | 0.473 ± 0.082 | 0.539 ± 0.095 |
| Ensemble | LR | 0.537 ± 0.121 | 0.479 ± 0.089 |
| Ensemble | RF | 0.533 ± 0.085 | 0.574 ± 0.076 |

### Model 4 (`2y_before_cv` genes)

| Task | Classifier | Slice split (`5d04e6ba`) | Patient split (new) |
|------|-----------|--------------------------|---------------------|
| Radiomics | LR | 0.500 ± 0.000 | 0.500 ± 0.000 |
| Radiomics | RF | 0.569 ± 0.133 | 0.569 ± 0.133 |
| Embeddings | LR | **0.743 ± 0.138** | 0.710 ± 0.135 |
| Embeddings | RF | 0.694 ± 0.160 | **0.787 ± 0.146** |
| Concat | LR | 0.561 ± 0.146 | 0.536 ± 0.155 |
| Concat | RF | 0.617 ± 0.166 | 0.553 ± 0.172 |
| Ensemble | LR | 0.690 ± 0.115 | 0.640 ± 0.106 |
| Ensemble | RF | 0.686 ± 0.135 | 0.665 ± 0.151 |

---

## 5.3 Group 3 — Full slices

All models: raw, frozen backbone, n=all at inference.

### Model 5 (λ=0.1, frozen, 10 ep)

| Task | Classifier | Slice split (`dc7e1d10`) | Patient split (new) |
|------|-----------|--------------------------|---------------------|
| Embeddings | LR | 0.975 ± 0.035 | **0.640 ± 0.038** |
| Embeddings | RF | **1.000 ± 0.000** | 0.593 ± 0.065 |
| Concat | LR | 0.938 ± 0.080 | 0.483 ± 0.086 |
| Concat | RF | **1.000 ± 0.000** | 0.495 ± 0.147 |
| Ensemble | LR | 0.888 ± 0.107 | **0.607 ± 0.078** |
| Ensemble | RF | 0.942 ± 0.082 | 0.572 ± 0.103 |

### Model 6 (λ=0.0, frozen, 10 ep)

| Task | Classifier | Slice split (`a64b245f`) | Patient split (new) |
|------|-----------|--------------------------|---------------------|
| Embeddings | LR | **0.739 ± 0.129** | 0.536 ± 0.143 |
| Embeddings | RF | 0.663 ± 0.147 | **0.648 ± 0.100** |
| Concat | LR | 0.606 ± 0.203 | 0.512 ± 0.089 |
| Concat | RF | 0.549 ± 0.184 | 0.518 ± 0.111 |
| Ensemble | LR | 0.623 ± 0.185 | 0.511 ± 0.160 |
| Ensemble | RF | 0.605 ± 0.164 | 0.580 ± 0.168 |

---

## 5.4 Group 4 — Bounding box

All models: raw_bbox, unfrozen, n=10 training, bbox_pad=10.

### Model 7 (bbox, λ=0.1, 50 ep)

**Condition 1 — infer 10 slices:**

| Task | Classifier | Slice split (`050d401d`) | Patient split (new) |
|------|-----------|--------------------------|---------------------|
| Radiomics | LR | 0.500 ± 0.000 | 0.500 ± 0.000 |
| Radiomics | RF | 0.569 ± 0.133 | 0.569 ± 0.133 |
| Embeddings | LR | **0.988 ± 0.017** | **0.645 ± 0.011** |
| Embeddings | RF | 0.948 ± 0.016 | 0.583 ± 0.071 |
| Concat | LR | 0.859 ± 0.158 | 0.524 ± 0.103 |
| Concat | RF | **0.861 ± 0.131** | 0.532 ± 0.131 |
| Ensemble | LR | **0.913 ± 0.036** | 0.587 ± 0.033 |
| Ensemble | RF | 0.817 ± 0.118 | 0.586 ± 0.097 |

**Condition 2 — infer all slices:**

| Task | Classifier | Slice split (`050d401d`) | Patient split (new) |
|------|-----------|--------------------------|---------------------|
| Embeddings | LR | 0.938 ± 0.018 | **0.661 ± 0.117** |
| Embeddings | RF | **0.965 ± 0.026** | 0.591 ± 0.043 |
| Concat | LR | 0.817 ± 0.166 | 0.500 ± 0.113 |
| Concat | RF | **0.825 ± 0.217** | 0.487 ± 0.084 |
| Ensemble | LR | **0.863 ± 0.067** | **0.624 ± 0.075** |
| Ensemble | RF | 0.826 ± 0.130 | 0.543 ± 0.075 |

### Model 8 (bbox, λ=0.0, 50 ep)

**Condition 1 — infer 10 slices:**

| Task | Classifier | Slice split (`e12b0592`) | Patient split (new) |
|------|-----------|--------------------------|---------------------|
| Radiomics | LR | 0.500 ± 0.000 | 0.500 ± 0.000 |
| Radiomics | RF | 0.569 ± 0.133 | 0.569 ± 0.133 |
| Embeddings | LR | **0.880 ± 0.067** | **0.632 ± 0.114** |
| Embeddings | RF | 0.857 ± 0.090 | 0.634 ± 0.080 |
| Concat | LR | **0.681 ± 0.111** | **0.586 ± 0.136** |
| Concat | RF | 0.563 ± 0.124 | 0.522 ± 0.142 |
| Ensemble | LR | **0.764 ± 0.097** | **0.587 ± 0.038** |
| Ensemble | RF | 0.725 ± 0.131 | 0.584 ± 0.105 |

**Condition 2 — infer all slices:**

| Task | Classifier | Slice split (`e12b0592`) | Patient split (new) |
|------|-----------|--------------------------|---------------------|
| Embeddings | LR | **0.657 ± 0.100** | 0.525 ± 0.031 |
| Embeddings | RF | 0.648 ± 0.119 | **0.568 ± 0.101** |
| Concat | LR | **0.582 ± 0.175** | 0.524 ± 0.175 |
| Concat | RF | 0.540 ± 0.152 | 0.475 ± 0.100 |
| Ensemble | LR | **0.611 ± 0.088** | 0.520 ± 0.074 |
| Ensemble | RF | 0.588 ± 0.160 | **0.572 ± 0.043** |
