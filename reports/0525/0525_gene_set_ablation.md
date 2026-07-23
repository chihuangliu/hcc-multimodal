
## Table of Contents

- [1. Task](#1-task)
- [2. Key findings](#2-key-findings)
- [3. Methodology](#3-methodology)
  - [3.1 Gene sets](#31-gene-sets)
  - [3.2 Contrastive pre-training](#32-contrastive-pre-training)
- [4. Training results](#4-training-results)
- [5. Downstream CV results](#5-downstream-cv-results)

# 1. Task

Ablate the gene set used for contrastive pre-training, comparing three subsets of the 40-gene RNA-seq panel on raw arterial-phase MRI (`Resections_with_rna/MRI_liver_arterial.nii.gz`). All other settings are identical to the 0525 raw λ=0.1 baseline (`6a1a1bdf`).

| Run | Gene set | # Genes | Model ID |
|-----|----------|---------|----------|
| Baseline (0525) | `all` | 40 | `6a1a1bdf` |
| Ablation A | `predefined_2y_cv` | 20 | `12e4ba6a` |
| Ablation B | `2y_before_cv` | 20 | `5d04e6ba` |

All runs: λ=0.1, raw MRI, ViT-B/32 unfrozen, 128-dim embeddings.

---

# 2. Key findings

**Training losses:**

| Gene set | Train loss (ep 50) | Best val loss (epoch) |
|----------|-------------------|----------------------|
| `all` (40 genes, baseline) | 1.330 | 1.327 (early) |
| `predefined_2y_cv` (20 genes) | 0.444 | 1.430 (ep 15) |
| `2y_before_cv` (20 genes) | 0.354 | 1.417 (ep 19) |

**Best embeddings AUC (in-CV):**

| Gene set | Best model | AUC ± std |
|----------|-----------|-----------|
| `all` (baseline) | RF | **0.798 ± 0.081** |
| `predefined_2y_cv` | RF | 0.585 ± 0.032 |
| `2y_before_cv` | LR | 0.743 ± 0.138 |

The full 40-gene set (`all`) yields the strongest image embeddings by a wide margin. The `2y_before_cv` 20-gene subset retains reasonable discriminative power (LR 0.743), while `predefined_2y_cv` drops sharply to 0.585. Both 20-gene subsets show a larger train/val gap during pre-training, suggesting the gene encoder overfits a smaller vocabulary and provides weaker supervision for the image encoder.

---

# 3. Methodology

## 3.1 Gene sets

| Key | Source | # Genes | Description |
|-----|--------|---------|-------------|
| `all` | `GENE_SET` | 40 | Full RNA-seq panel used across the project |
| `predefined_2y_cv` | `PREDEFINED_HCC_2Y_CV_GENES` | 20 | Pre-selected genes associated with 2-year RFS in the CV literature |
| `2y_before_cv` | `RNA_2Y_BEFORE_CV_GENES` | 20 | Genes expressed before cirrhosis/CV event, associated with 2-year RFS |

## 3.2 Contrastive pre-training

All three runs share:

| Parameter | Value |
|-----------|-------|
| Backbone | ViT-B/32 (unfrozen) |
| Embed dim | 128 per modality |
| Gene hidden dim | 256 |
| Temperature τ | 0.07 |
| λ (reg weight) | 0.1 |
| reg_mode | per_modality |
| MRI type | `raw` (`Resections_with_rna`) |
| Slices per patient | 10 (sagittal axis=0) |
| Image size | 224 × 224 |
| Epochs | 50 |
| Batch size | 32 |
| Optimiser | AdamW (lr=1e-4, wd=1e-4) |
| LR schedule | Cosine annealing (T_max=50) |
| Val split | 10% stratified hold-out |
| Checkpoint | Best validation loss |
| Seed | 42 |

---

# 4. Training results

Epoch 50 train loss and best (lowest) validation loss across 50 epochs.

| Gene set | Model ID | Train loss (ep 50) | Best val loss | Best val epoch |
|----------|----------|-------------------|--------------|---------------|
| `all` (baseline) | `6a1a1bdf` | 1.330 | 1.327 | early |
| `predefined_2y_cv` | `12e4ba6a` | 0.444 | 1.430 | 15 |
| `2y_before_cv` | `5d04e6ba` | 0.354 | 1.417 | 19 |

The 20-gene runs show a classic train/val gap: training loss descends much further (0.35–0.44 vs 1.33) while validation loss is higher, indicating the gene encoder can more easily overfit a smaller gene vocabulary. The `all` run converges smoothly with train ≈ val, suggesting better-calibrated supervision.

---

# 5. Downstream CV results

3-fold stratified CV, SelectKBest(f_classif, k=100) fitted in-CV only. ROC-AUC mean ± std.

Radiomics baseline is gene-set-independent (same result for all runs).

| Task | Model | `all` λ=0.1 (baseline) | `predefined_2y_cv` | `2y_before_cv` |
|------|-------|----------------------|--------------------|----------------|
| Radiomics | LR | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.500 ± 0.000 |
| Radiomics | RF | 0.569 ± 0.133 | 0.569 ± 0.133 | 0.569 ± 0.133 |
| Embeddings | LR | 0.752 ± 0.062 | 0.495 ± 0.145 | **0.743 ± 0.138** |
| Embeddings | RF | **0.798 ± 0.081** | 0.585 ± 0.032 | 0.694 ± 0.160 |
| Concat | LR | 0.516 ± 0.092 | 0.450 ± 0.019 | 0.561 ± 0.146 |
| Concat | RF | 0.489 ± 0.096 | 0.473 ± 0.082 | 0.617 ± 0.166 |
| Ensemble | LR | 0.648 ± 0.096 | 0.537 ± 0.121 | 0.690 ± 0.115 |
| Ensemble | RF | 0.696 ± 0.078 | 0.533 ± 0.085 | 0.686 ± 0.135 |

Model IDs: `all`=`6a1a1bdf`, `predefined_2y_cv`=`12e4ba6a`, `2y_before_cv`=`5d04e6ba`.
