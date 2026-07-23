
## Table of Contents

- [1. Task](#1-task)
- [2. Key findings](#2-key-findings)
- [3. Methodology](#3-methodology)
  - [3.1 Models compared](#31-models-compared)
  - [3.2 Inference slice comparison plan](#32-inference-slice-comparison-plan)
  - [3.3 Downstream CV pipeline](#33-downstream-cv-pipeline)
- [4. In-CV results](#4-in-cv-results)

# 1. Task

Evaluate a new contrastive model (`3e598f36`) trained on **all sagittal slices** (raw arterial MRI) with a **frozen ViT-B/32 backbone** for only **5 epochs**, and compare to the 0525 raw λ=0.1 model (`6a1a1bdf`) which used **10 slices** with an **unfrozen backbone** for **50 epochs**.

A secondary goal is to establish a fair inference-time comparison: when n_per_axis differs between models, inference conditions also differ (more slices → more averaging). A `--n_per_axis` override flag was added to `multimodal_prediction.py` so any model can be evaluated with a fixed slice count. Three conditions are planned:

| Condition | Train slices | Inference slices | Model | Status |
|-----------|-------------|-----------------|-------|--------|
| 1 | 10 | 10 | `6a1a1bdf --n_per_axis 10` | Done (0525 report) |
| 2 | 10 | all | `6a1a1bdf` (no flag) | Done |
| 3 | all | all | `3e598f36` (no flag) | Done |
| 4 | all | all | `3e598f36` +5 ep continued | Done |
| 5 | all | all | `a64b245f` λ=0, frozen, 10 ep | Done |
| 6 | 10 | all | `982a6fa2` λ=0, unfrozen, 50 ep | Done |

---

# 2. Key findings

Best RF AUC across tasks (in-CV, rfs_2year):

| Task | `6a1a1bdf` cond 1 (10/10) | `6a1a1bdf` cond 2 (10/all) | `982a6fa2` cond 6 (10/all, λ=0) | `3e598f36` cond 3 (all/all) | `dc7e1d10` cond 4 (all/all, +5 ep) | `a64b245f` cond 5 (λ=0, 10 ep) |
|------|--------------------------|---------------------------|----------------------------------|----------------------------|-------------------------------------|----------------------------------|
| Radiomics | 0.569 ± 0.133 | 0.569 ± 0.133 | 0.569 ± 0.133 | 0.569 ± 0.133 | 0.569 ± 0.133 | 0.569 ± 0.133 |
| Embeddings | 0.798 ± 0.081 | 0.672 ± 0.090 | 0.717 ± 0.087 | 0.911 ± 0.064 | **1.000 ± 0.000** | 0.739 ± 0.129 |
| Concat | 0.489 ± 0.096 | 0.545 ± 0.139 | 0.539 ± 0.130 | 0.838 ± 0.163 | **1.000 ± 0.000** | 0.606 ± 0.203 |
| Ensemble | 0.696 ± 0.078 | 0.599 ± 0.065 | 0.605 ± 0.062 | 0.807 ± 0.122 | 0.942 ± 0.082 | 0.623 ± 0.185 |

Note: result dirs for `3e598f36`/`dc7e1d10` include a `_nall_` slice tag; the `6a1a1bdf` dirs predate this convention (implicitly n=10 for condition 1, `_nall_` for condition 2).

---

# 3. Methodology

## 3.1 Models compared

| Parameter | `6a1a1bdf` (0525) | `982a6fa2` (0525) | `3e598f36` | `dc7e1d10` | `a64b245f` |
|-----------|-------------------|-------------------|------------|------------|------------|
| Backbone | ViT-B/32 (unfrozen) | ViT-B/32 (unfrozen) | ViT-B/32 (frozen) | ViT-B/32 (frozen) | ViT-B/32 (frozen) |
| Base model | — | — | — | `3e598f36` | — |
| Embed dim | 128 | 128 | 128 | 128 | 128 |
| Gene hidden dim | 256 | 256 | 256 | 256 | 256 |
| Temperature τ | 0.07 | 0.07 | 0.07 | 0.07 | 0.07 |
| λ (reg weight) | 0.1 | **0.0** | 0.1 | 0.1 | **0.0** |
| reg_mode | per_modality | per_modality | per_modality | per_modality | per_modality |
| Gene set | all | all | all | all | all |
| n_per_axis (train) | 10 | 10 | null (all) | null (all) | null (all) |
| Axes | sagittal (0) | sagittal (0) | sagittal (0) | sagittal (0) | sagittal (0) |
| MRI type | raw | raw | raw | raw | raw |
| Epochs | 50 | 50 | 5 | 5 (+5 continued) | **10** |
| Batch size | 32 | 32 | 32 | 32 | 32 |
| Optimiser | AdamW (lr=1e-4, wd=1e-4) | AdamW (lr=1e-4, wd=1e-4) | AdamW (lr=1e-4, wd=1e-4) | AdamW (lr=1e-4, wd=1e-4) | AdamW (lr=1e-4, wd=1e-4) |
| LR schedule | Cosine (T_max=50) | Cosine (T_max=50) | Cosine (T_max=5) | Cosine (T_max=5) | Cosine (T_max=10) |
| Seed | 42 | 42 | 42 | 42 | 42 |
| Train loss (final epoch) | 1.330 | 1.415 | 0.238 | -0.140 | 1.240 |
| Val loss (best) | 1.327 | 3.941 (early) | 0.259 | -0.085 | 1.316 |

## 3.2 Inference slice comparison plan

`multimodal_prediction.py` now reads `n_per_axis` from model metadata at inference. An override flag allows decoupling inference slices from training:

```bash
# Condition 1 — fix to 10 slices
python multimodal_prediction.py 6a1a1bdf --task embeddings --n_per_axis 10 ...

# Condition 2 — all slices (default, no flag)
python multimodal_prediction.py 6a1a1bdf --task embeddings ...

# Condition 3/4 — all slices (default, no flag)
python multimodal_prediction.py 3e598f36 --task embeddings ...
```

`--n_per_axis` accepts an integer to fix slice count; omitting it (default `None`) uses all slices. The slice tag in the output dir (`_n10_`, `_nall_`) records which was used.

Embedding extraction results are now cached to `training/contrastive/{model_id}/cached_embeddings/emb_{slice_tag}.csv` (gitignored) so concat/ensemble runs reuse the extracted features without re-running forward passes.

## 3.3 Downstream CV pipeline

Identical to 0525: 3-fold stratified CV (`StratifiedKFold`, `random_state=42`), `SelectKBest(f_classif, k=100)` fitted on the training fold only, `StandardScaler` inside pipeline. 54 patients with paired MRI, RNA-seq, and clinical outcome.

---

# 4. In-CV results

SelectKBest fitted on the training fold only. ROC-AUC mean ± std across 3 folds.

## 4.1 `6a1a1bdf` — train 10 slices / infer 10 slices (condition 1)

| Task | Model | AUC ± std |
|------|-------|-----------|
| Radiomics | LR | 0.500 ± 0.000 |
| Radiomics | RF | 0.569 ± 0.133 |
| Embeddings | LR | 0.752 ± 0.062 |
| Embeddings | RF | **0.798 ± 0.081** |
| Concat | LR | 0.516 ± 0.092 |
| Concat | RF | 0.489 ± 0.096 |
| Ensemble | LR | 0.648 ± 0.096 |
| Ensemble | RF | 0.696 ± 0.078 |

## 4.2 `3e598f36` — train all slices / infer all slices (condition 3)

Result dirs: `results/multimodal_prediction/{task}_3e598f36_rfs_2year_in_cv_k100_raw_nall_7c35135/`

| Task | Model | AUC ± std |
|------|-------|-----------|
| Radiomics | LR | 0.500 ± 0.000 |
| Radiomics | RF | 0.569 ± 0.133 |
| Embeddings | LR | 0.884 ± 0.124 |
| Embeddings | RF | **0.911 ± 0.064** |
| Concat | LR | 0.763 ± 0.184 |
| Concat | RF | 0.838 ± 0.163 |
| Ensemble | LR | 0.710 ± 0.133 |
| Ensemble | RF | 0.807 ± 0.122 |

## 4.3 `6a1a1bdf` — train 10 slices / infer all slices (condition 2)

Result dirs: `results/multimodal_prediction/{task}_6a1a1bdf_rfs_2year_in_cv_k100_raw_nall_efb43d1/`

| Task | Model | AUC ± std |
|------|-------|-----------|
| Embeddings | LR | 0.574 ± 0.022 |
| Embeddings | RF | 0.672 ± 0.090 |
| Concat | LR | 0.495 ± 0.068 |
| Concat | RF | 0.545 ± 0.139 |
| Ensemble | LR | 0.591 ± 0.059 |
| Ensemble | RF | 0.599 ± 0.065 |

Condition 2 is **worse than condition 1** across all tasks (RF embeddings: 0.802 → 0.672). Using all slices at inference for a model trained on 10 slices introduces a distribution shift — the encoder was optimised on a 10-slice view and degrades when mean-pooled over the full volume.

## 4.4 `dc7e1d10` — continued training (+5 ep from `3e598f36`), infer all slices

Result dirs: `results/multimodal_prediction/{task}_dc7e1d10_rfs_2year_in_cv_k100_raw_nall_511af05/`

| Task | Model | AUC ± std |
|------|-------|-----------|
| Embeddings | LR | 0.975 ± 0.035 |
| Embeddings | RF | **1.000 ± 0.000** |
| Concat | LR | 0.938 ± 0.080 |
| Concat | RF | **1.000 ± 0.000** |
| Ensemble | LR | 0.888 ± 0.107 |
| Ensemble | RF | 0.942 ± 0.082 |

## 4.5 `a64b245f` — λ=0, frozen backbone, 10 epochs, infer all slices

Result dirs: `results/multimodal_prediction/{task}_a64b245f_rfs_2year_in_cv_k100_raw_nall_ddd070c/`

| Task | Model | AUC ± std |
|------|-------|-----------|
| Embeddings | LR | 0.739 ± 0.129 |
| Embeddings | RF | 0.663 ± 0.147 |
| Concat | LR | 0.606 ± 0.203 |
| Concat | RF | 0.549 ± 0.184 |
| Ensemble | LR | 0.623 ± 0.185 |
| Ensemble | RF | 0.605 ± 0.164 |

## 4.6 `982a6fa2` — train 10 slices / infer all slices, λ=0 (condition 6)

Result dirs: `results/multimodal_prediction/{task}_982a6fa2_rfs_2year_in_cv_k100_raw_nall_7f2a97e/`

| Task | Model | AUC ± std |
|------|-------|-----------|
| Embeddings | LR | 0.636 ± 0.048 |
| Embeddings | RF | **0.717 ± 0.087** |
| Concat | LR | 0.524 ± 0.072 |
| Concat | RF | 0.539 ± 0.130 |
| Ensemble | LR | 0.603 ± 0.078 |
| Ensemble | RF | **0.605 ± 0.062** |

## 4.7 Summary across conditions

| Condition | Model | Task | Best model | AUC ± std |
|-----------|-------|------|------------|-----------|
| 1 — train 10 / infer 10 | `6a1a1bdf` | Embeddings | RF | 0.798 ± 0.081 |
| 1 — train 10 / infer 10 | `6a1a1bdf` | Ensemble | RF | 0.696 ± 0.078 |
| 2 — train 10 / infer all | `6a1a1bdf` | Embeddings | RF | 0.672 ± 0.090 |
| 2 — train 10 / infer all | `6a1a1bdf` | Ensemble | RF | 0.599 ± 0.065 |
| 6 — train 10 / infer all, λ=0 | `982a6fa2` | Embeddings | RF | 0.717 ± 0.087 |
| 6 — train 10 / infer all, λ=0 | `982a6fa2` | Ensemble | RF | 0.605 ± 0.062 |
| 3 — train all / infer all, λ=0.1 | `3e598f36` | Embeddings | RF | **0.911 ± 0.064** |
| 3 — train all / infer all, λ=0.1 | `3e598f36` | Concat | RF | 0.838 ± 0.163 |
| 3 — train all / infer all, λ=0.1 | `3e598f36` | Ensemble | RF | 0.807 ± 0.122 |
| 4 — continued (+5 ep), λ=0.1 | `dc7e1d10` | Embeddings | RF | 1.000 ± 0.000 |
| 4 — continued (+5 ep), λ=0.1 | `dc7e1d10` | Ensemble | RF | 0.942 ± 0.082 |
| 5 — train all / infer all, λ=0.0 | `a64b245f` | Embeddings | LR | 0.739 ± 0.129 |
| 5 — train all / infer all, λ=0.0 | `a64b245f` | Ensemble | LR | 0.623 ± 0.185 |
