# 0525 Summary

## 1. Key Findings

### 1.1 Raw image + resampling vs Preprocessed image (best embeddings AUC, in-CV)

| MRI type | λ | Best model | AUC ± std |
|----------|---|------------|-----------|
| Preprocessed (0518) | 0.1 | RF | 0.706 ± 0.093 |
| Preprocessed (0518) | 0.0 | RF | 0.752 ± 0.111 |
| Raw (0525) | 0.1 | RF | **0.798 ± 0.081** |
| Raw (0525) | 0.0 | LR | 0.706 ± 0.099 |

### 1.2 Gene set ablation (best embeddings AUC, in-CV, all λ=0.1 raw)

| Gene set | # Genes | Best model | AUC ± std |
|----------|---------|------------|-----------|
| `all` (baseline) | 40 | RF | **0.798 ± 0.081** |
| `2y_before_cv` | 20 | LR | 0.743 ± 0.138 |
| `predefined_2y_cv` | 20 | RF | 0.585 ± 0.032 |

### 1.3 Train on 10 slices vs all slices (best embeddings AUC, in-CV, raw MRI)

| Train slices | λ | Backbone | Epochs | Infer slices | Best model | AUC ± std |
|-------------|---|----------|--------|--------------|------------|-----------|
| 10 | 0.1 | unfrozen | 50 | 10 | RF | 0.798 ± 0.081 |
| 10 | 0.1 | unfrozen | 50 | all | RF | 0.672 ± 0.090 |
| 10 | 0.0 | unfrozen | 50 | all | RF | 0.717 ± 0.087 |
| all | 0.1 | frozen | 5 | all | RF | 0.911 ± 0.064 |
| all | 0.1 | frozen | 10 (+5 continued) | all | RF | **1.000 ± 0.000** |
| all | 0.0 | frozen | 10 | all | LR | 0.739 ± 0.129 |

### 1.4 Bounding box combinations (best embeddings AUC, in-CV)

| BBox | λ | Infer slices | Best model | AUC ± std |
|------|---|--------------|------------|-----------|
| No | 0.1 | 10 | RF | 0.798 ± 0.081 |
| No | 0.1 | all | RF | 0.672 ± 0.090 |
| No | 0.0 | 10 | LR | 0.706 ± 0.099 |
| No | 0.0 | all | RF | 0.717 ± 0.087 |
| Yes | 0.1 | 10 | LR | **0.988 ± 0.017** |
| Yes | 0.1 | all | RF | 0.965 ± 0.026 |
| Yes | 0.0 | 10 | LR | 0.880 ± 0.067 |
| Yes | 0.0 | all | LR | 0.657 ± 0.100 |

---

## 2. Raw Image + Resampling vs Preprocessed Image

**Task**: Replicate 0518 in-CV experiments (λ=0.1 and λ=0) using raw arterial-phase MRI (`Resections_with_rna/MRI_liver_arterial.nii.gz`) resampled to 1×1×3 mm, with no further intensity pre-processing beyond per-slice normalisation.

| Task | Model | Preprocessed λ=0.1 | Preprocessed λ=0 | Raw λ=0.1 | Raw λ=0 |
|------|-------|-------------------|-----------------|-----------|---------|
| Radiomics | LR | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.500 ± 0.000 |
| Radiomics | RF | 0.569 ± 0.133 | 0.569 ± 0.133 | 0.569 ± 0.133 | 0.569 ± 0.133 |
| Embeddings | LR | 0.623 ± 0.129 | 0.703 ± 0.103 | 0.752 ± 0.062 | **0.706 ± 0.099** |
| Embeddings | RF | 0.706 ± 0.093 | **0.752 ± 0.111** | **0.798 ± 0.081** | 0.671 ± 0.065 |
| Concat | LR | — | — | 0.516 ± 0.092 | 0.529 ± 0.085 |
| Concat | RF | — | — | 0.489 ± 0.096 | 0.559 ± 0.144 |
| Ensemble | LR | — | — | 0.648 ± 0.096 | 0.578 ± 0.029 |
| Ensemble | RF | — | — | 0.696 ± 0.078 | 0.595 ± 0.063 |

Model IDs: Raw λ=0.1 = `6a1a1bdf`, Raw λ=0 = `982a6fa2`.

---

## 3. Gene Set Ablation Study

**Task**: Ablate the gene set used for contrastive pre-training. All runs use raw MRI, λ=0.1, ViT-B/32 unfrozen, 128-dim embeddings.

**Training losses:**

| Gene set | # Genes | Model ID | Train loss (ep 50) | Best val loss (epoch) |
|----------|---------|----------|-------------------|-----------------------|
| `all` (baseline) | 40 | `6a1a1bdf` | 1.330 | 1.327 (early) |
| `predefined_2y_cv` | 20 | `12e4ba6a` | 0.444 | 1.430 (ep 15) |
| `2y_before_cv` | 20 | `5d04e6ba` | 0.354 | 1.417 (ep 19) |

**Downstream CV (ROC-AUC mean ± std, 3-fold in-CV):**

| Task | Model | `all` (40 genes) | `predefined_2y_cv` (20) | `2y_before_cv` (20) |
|------|-------|-----------------|------------------------|---------------------|
| Radiomics | RF | 0.569 ± 0.133 | 0.569 ± 0.133 | 0.569 ± 0.133 |
| Embeddings | LR | 0.752 ± 0.062 | 0.495 ± 0.145 | **0.743 ± 0.138** |
| Embeddings | RF | **0.798 ± 0.081** | 0.585 ± 0.032 | 0.694 ± 0.160 |
| Concat | LR | 0.516 ± 0.092 | 0.450 ± 0.019 | 0.561 ± 0.146 |
| Concat | RF | 0.489 ± 0.096 | 0.473 ± 0.082 | 0.617 ± 0.166 |
| Ensemble | LR | 0.648 ± 0.096 | 0.537 ± 0.121 | 0.690 ± 0.115 |
| Ensemble | RF | 0.696 ± 0.078 | 0.533 ± 0.085 | 0.686 ± 0.135 |

---

## 4. Bounding Boxes

**Task**: Replicate 0525 raw experiments with tumour bounding-box cropped MRI (`raw_bbox`). Bbox computed from `hcc_seg_reg*.nii.gz`, resampled to 1×1×3 mm, padded by 10 voxels per face. Downstream CV evaluated at two inference conditions: 10 slices (condition 1) and all slices within bbox (condition 2).

Model IDs: BBox λ=0.1 = `050d401d`, BBox λ=0 = `e12b0592`.

**Condition 1 — infer 10 slices:**

| Task | Model | BBox λ=0.1 | BBox λ=0 | Raw λ=0.1 | Raw λ=0 |
|------|-------|-----------|---------|-----------|--------|
| Radiomics | RF | 0.569 ± 0.133 | 0.569 ± 0.133 | 0.569 ± 0.133 | 0.569 ± 0.133 |
| Embeddings | LR | **0.988 ± 0.017** | **0.880 ± 0.067** | 0.752 ± 0.062 | **0.706 ± 0.099** |
| Embeddings | RF | 0.948 ± 0.016 | 0.857 ± 0.090 | **0.798 ± 0.081** | 0.671 ± 0.065 |
| Concat | LR | 0.859 ± 0.158 | **0.681 ± 0.111** | 0.516 ± 0.092 | 0.529 ± 0.085 |
| Concat | RF | **0.861 ± 0.131** | 0.563 ± 0.124 | 0.489 ± 0.096 | 0.559 ± 0.144 |
| Ensemble | LR | **0.913 ± 0.036** | **0.764 ± 0.097** | 0.648 ± 0.096 | 0.578 ± 0.029 |
| Ensemble | RF | 0.817 ± 0.118 | 0.725 ± 0.131 | 0.696 ± 0.078 | 0.595 ± 0.063 |

**Condition 2 — infer all slices:**

| Task | Model | BBox λ=0.1 | BBox λ=0 | Raw λ=0.1 |
|------|-------|-----------|---------|-----------|
| Embeddings | LR | 0.938 ± 0.018 | **0.657 ± 0.100** | 0.574 ± 0.022 |
| Embeddings | RF | **0.965 ± 0.026** | 0.648 ± 0.119 | 0.672 ± 0.090 |
| Concat | LR | 0.817 ± 0.166 | **0.582 ± 0.175** | 0.495 ± 0.068 |
| Concat | RF | **0.825 ± 0.217** | 0.540 ± 0.152 | 0.545 ± 0.139 |
| Ensemble | LR | **0.863 ± 0.067** | **0.611 ± 0.088** | 0.591 ± 0.059 |
| Ensemble | RF | 0.826 ± 0.130 | 0.588 ± 0.160 | 0.599 ± 0.065 |

---

## 5. Train on Full Slices

**Task**: Evaluate models trained on all sagittal slices (raw arterial MRI) vs the 0525 baseline trained on 10 slices. An inference-time `--n_per_axis` override flag allows fixed or full-slice evaluation independently of training configuration.

Models and conditions:

| Model ID | Train slices | λ | Backbone | Epochs |
|----------|-------------|---|----------|--------|
| `6a1a1bdf` | 10 | 0.1 | unfrozen | 50 |
| `3e598f36` | all | 0.1 | frozen | 5 |
| `dc7e1d10` | all | 0.1 | frozen | 5+5 (continued) |
| `a64b245f` | all | 0.0 | frozen | 10 |

**Best RF AUC per task across all conditions (in-CV):**

| Task | `6a1a1bdf` cond 1 (10/10) | `6a1a1bdf` cond 2 (10/all) | `3e598f36` cond 3 (all/all) | `dc7e1d10` cond 4 (all/all +5ep) | `a64b245f` cond 5 (λ=0, all/all) |
|------|--------------------------|---------------------------|----------------------------|----------------------------------|----------------------------------|
| Radiomics | 0.569 ± 0.133 | 0.569 ± 0.133 | 0.569 ± 0.133 | 0.569 ± 0.133 | 0.569 ± 0.133 |
| Embeddings | 0.798 ± 0.081 | 0.672 ± 0.090 | 0.911 ± 0.064 | **1.000 ± 0.000** | 0.663 ± 0.147 |
| Concat | 0.489 ± 0.096 | 0.545 ± 0.139 | 0.838 ± 0.163 | **1.000 ± 0.000** | 0.549 ± 0.184 |
| Ensemble | 0.696 ± 0.078 | 0.599 ± 0.065 | 0.807 ± 0.122 | 0.942 ± 0.082 | 0.605 ± 0.164 |
