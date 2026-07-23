# HCC Multimodal — Baseline Results (2026-04-20)

## Conclusion

Best results per modality across all pipelines:

### Logistic Regression

| Modality | Best preprocess | Target | AUC |
|----------|-------------|--------|-----|
| Clinical | PCA | OS_central_event | 0.597 ± 0.019 |
| Radiomic | PCA | death | 0.642 ± 0.069 |
| RNA-seq | log2(CPM+1) + SelectKBest(500) + PCA | death | **0.670 ± 0.130** |

### Random Forest

| Modality | Best preprocess | Target | AUC |
|----------|-------------|--------|-----|
| Clinical | PCA | OS_central_event | 0.669 ± 0.131 |
| Radiomic | PCA | death | **0.670 ± 0.101** |
| RNA-seq | PCA (raw counts) | death | 0.611 ± 0.197 |


---

## Methodology

**Task:** Binary classification of post-resection outcome in HCC patients.

**Targets:**
- `OS_central_event` — overall survival event flag from the clinical CSV (`data/Clinical/2025_Nov_18_ICL_Resection_Clinical_Outcome_soramic_format.csv`)
- `death` — binary death label from the radiomics CSV (`data/Radiomics/radiomic_cluster.csv`)

**Models:** Logistic Regression and Random Forest.

**Evaluation:** Stratified 3-fold CV (outer loop). AUC (ROC) is the primary metric.  
**Hyperparameter tuning:** Optional inner 3-fold CV grid search over `C ∈ {1, 10, 100}` + `l1_ratio ∈ {0.5, 1.0}` for LR, and `max_depth ∈ {3, 5, 10, None}` + `min_samples_leaf ∈ {1, 3, 5}` for RF.

### Standard pipeline

- Continuous: median imputation → StandardScaler
- Categorical: constant imputation → OneHotEncoder
- Ordinal: mode imputation → OrdinalEncoder
- All modalities: PCA retaining 90% variance

### Additional processing pipeline (Radiomic & RNA-seq)

`notebooks/baselines/radiomic_baseline_v2.ipynb`, `notebooks/baselines/rna_baseline_v2.ipynb`

Four pipeline variants are compared against the standard pipeline:

| Variant | SelectKBest | PCA | LR regularisation |
|---------|-------------|-----|-------------------|
| 1. Original | — | 90% var | C=1.0 |
| 2. KBest + PCA | k features | 90% var | C=1.0 |
| 3. No PCA, small C | — | — | C=0.1 (strong L1) |
| 4. KBest + no PCA, small C | k features | — | C=0.1 (strong L1) |

**RNA-seq specific preprocessing** (applied before all four variants):
- Low-expression gene filtering: keep genes with count > 10 in ≥ 20% of samples (50,986 → 21,241 genes)
- log2(CPM + 1) normalisation: corrects library-size bias (range 234 K–159 M reads) and compresses dynamic range

**Configuration:** RNA SelectKBest k=500; Radiomic SelectKBest k=100; LR solver=saga, l1_ratio=1.0, max_iter=5000; RF 100 trees.

---

## 1. Clinical Baseline

`notebooks/baselines/pure_clinical_baseline.ipynb`

**Features:** 11 clinical variables — Age, Sex, BCLC Stage, Max. Tumour Diameter, HCC Etiology (HBV/HCV/Alcohol/NASH), Child-Pugh, Vascular Invasion, Histological Grade.  
**Target:** `OS_central_event`  
**Dataset:** n=64, 27 events (42%)

### Without grid search

| Model | AUC mean ± std | Acc mean ± std |
|-------|---------------|----------------|
| LR | 0.597 ± 0.019 | 0.562 ± 0.028 |
| RF | 0.669 ± 0.131 | 0.563 ± 0.079 |

### With grid search

| Model | AUC mean ± std | Acc mean ± std |
|-------|---------------|----------------|
| LR | 0.579 ± 0.018 | 0.546 ± 0.032 |
| RF | 0.639 ± 0.102 | 0.562 ± 0.062 |

---

## 2. Radiomic Baseline

`notebooks/baselines/radiomic_baseline.ipynb`

**Features:** 446 pre-computed radiomic features (texture, shape, intensity; arterial + delayed phase).  
**Dimensionality:** PCA reduces to ~39–42 components at 90% variance threshold.  
**Dataset overlap:** 60 patients with both radiomic and clinical data.

### Without grid search

| Experiment | n | Model | AUC mean ± std | Acc mean ± std |
|------------|---|-------|---------------|----------------|
| Radiomic → death | 60 | LR | 0.642 ± 0.069 | 0.700 ± 0.071 |
| Radiomic → death | 60 | RF | 0.670 ± 0.101 | 0.667 ± 0.024 |
| Radiomic → OS_central_event | 56 | LR | 0.359 ± 0.141 | 0.443 ± 0.125 |
| Radiomic → OS_central_event | 56 | RF | 0.441 ± 0.076 | 0.517 ± 0.055 |
| Clinical+Radiomic → OS_central_event | 56 | LR | 0.513 ± 0.044 | 0.551 ± 0.114 |
| Clinical+Radiomic → OS_central_event | 56 | RF | 0.512 ± 0.169 | 0.588 ± 0.075 |

### With grid search

| Experiment | Model | AUC mean ± std | Acc mean ± std |
|------------|-------|---------------|----------------|
| Radiomic → death | LR | 0.628 ± 0.044 | 0.633 ± 0.024 |
| Radiomic → death | RF | 0.672 ± 0.103 | 0.633 ± 0.047 |
| Radiomic → OS_central_event | LR | 0.346 ± 0.156 | 0.443 ± 0.125 |
| Radiomic → OS_central_event | RF | 0.387 ± 0.014 | 0.517 ± 0.055 |
| Clinical+Radiomic → OS_central_event | LR | 0.512 ± 0.059 | 0.569 ± 0.088 |
| Clinical+Radiomic → OS_central_event | RF | 0.504 ± 0.146 | 0.588 ± 0.075 |

### Additional processing

`notebooks/baselines/radiomic_baseline_v2.ipynb` — SelectKBest k=100, LR_C_SMALL=0.1

| Experiment | n | Model | Pipeline | AUC mean ± std | Acc mean ± std |
|------------|---|-------|----------|---------------|----------------|
| Radiomic → death | 60 | LR | KBest(100) + PCA | 0.531 ± 0.113 | 0.617 ± 0.094 |
| Radiomic → death | 60 | RF | KBest(100) + PCA | 0.601 ± 0.108 | 0.633 ± 0.024 |
| Radiomic → OS_central_event | 56 | LR | KBest(100) + PCA | 0.382 ± 0.050 | 0.408 ± 0.092 |
| Radiomic → OS_central_event | 56 | RF | KBest(100) + PCA | 0.453 ± 0.052 | 0.481 ± 0.034 |
| Radiomic → death | 60 | LR (small C) | no PCA | 0.576 ± 0.057 | 0.600 ± 0.000 |
| Radiomic → OS_central_event | 56 | LR (small C) | no PCA | 0.500 ± 0.000 | 0.625 ± 0.010 |
| Radiomic → death | 60 | LR (small C) | KBest(100), no PCA | 0.576 ± 0.057 | 0.600 ± 0.000 |
| Radiomic → OS_central_event | 56 | LR (small C) | KBest(100), no PCA | 0.500 ± 0.000 | 0.625 ± 0.010 |

**Note:** All additional-processed variants underperform the original PCA-only pipeline. With only 446 features, PCA is already an effective dimensionality reduction step; SelectKBest discards informative features. The strong L1 penalty (C=0.1) degenerates on `OS_central_event` (AUC=0.500), and Exp 3 and Exp 4 produce identical results, indicating L1 already zeroes the same features that SelectKBest removes.

---

## 3. RNA-seq Baseline

`notebooks/baselines/rna_baseline.ipynb`

**Features:** ~50,986 gene expression counts (bulk RNA-seq; rows = genes, columns = patients). Transposed and deduplicated before modelling.  
**Dimensionality:** PCA reduces high-dimensional gene space substantially before classification.  
**Dataset overlap:** 60 patients with RNA, clinical, and radiomic data.

### Without grid search

| Experiment | n | Model | AUC mean ± std | Acc mean ± std |
|------------|---|-------|---------------|----------------|
| RNA → death | 60 | LR | 0.649 ± 0.094 | 0.617 ± 0.062 |
| RNA → death | 60 | RF | 0.611 ± 0.197 | 0.600 ± 0.082 |
| RNA → OS_central_event | 56 | LR | 0.504 ± 0.171 | 0.551 ± 0.114 |
| RNA → OS_central_event | 56 | RF | 0.474 ± 0.060 | 0.552 ± 0.079 |
| RNA+Clinical → death | 60 | LR | 0.653 ± 0.089 | 0.617 ± 0.062 |
| RNA+Clinical → death | 60 | RF | 0.615 ± 0.188 | 0.567 ± 0.062 |
| RNA+Clinical → OS_central_event | 56 | LR | 0.504 ± 0.171 | 0.551 ± 0.114 |
| RNA+Clinical → OS_central_event | 56 | RF | 0.416 ± 0.022 | 0.534 ± 0.063 |

### With grid search

| Experiment | Model | AUC mean ± std | Acc mean ± std |
|------------|-------|---------------|----------------|
| RNA → death | LR | 0.653 ± 0.085 | 0.617 ± 0.062 |
| RNA → death | RF | 0.622 ± 0.170 | 0.567 ± 0.085 |
| RNA → OS_central_event | LR | 0.496 ± 0.176 | 0.533 ± 0.104 |
| RNA → OS_central_event | RF | 0.468 ± 0.044 | 0.568 ± 0.142 |
| RNA+Clinical → death | LR | 0.653 ± 0.085 | 0.617 ± 0.062 |
| RNA+Clinical → death | RF | 0.625 ± 0.179 | 0.550 ± 0.071 |
| RNA+Clinical → OS_central_event | LR | 0.500 ± 0.176 | 0.551 ± 0.114 |
| RNA+Clinical → OS_central_event | RF | 0.440 ± 0.026 | 0.568 ± 0.129 |

### Additional processing

`notebooks/baselines/rna_baseline_v2.ipynb` — log2(CPM+1) + low-expression filtering → 21,241 genes; SelectKBest k=500, LR_C_SMALL=0.1

| Experiment | n | Model | Pipeline | AUC mean ± std | Acc mean ± std |
|------------|---|-------|----------|---------------|----------------|
| RNA → death | 60 | LR | KBest(500) + PCA | **0.670 ± 0.130** | 0.633 ± 0.170 |
| RNA → death | 60 | RF | KBest(500) + PCA | 0.609 ± 0.112 | 0.633 ± 0.131 |
| RNA → OS_central_event | 56 | LR | KBest(500) + PCA | **0.586 ± 0.100** | 0.481 ± 0.111 |
| RNA → OS_central_event | 56 | RF | KBest(500) + PCA | 0.421 ± 0.099 | 0.497 ± 0.173 |
| RNA → death | 60 | LR (small C) | KBest(500), no PCA | 0.620 ± 0.162 | 0.633 ± 0.024 |
| RNA → OS_central_event | 56 | LR (small C) | KBest(500), no PCA | 0.437 ± 0.090 | 0.625 ± 0.010 |

**Note:** Log2(CPM+1) normalisation corrects library-size bias (234 K–159 M reads across samples, a 670× range). Combined with SelectKBest, LR on `death` matches the radiomic best (AUC 0.670) and `OS_central_event` improves substantially over the raw-count baseline (+0.082). The no-PCA L1 variant performs reasonably on `death` (0.620) but degrades on `OS_central_event`, where the signal is weaker.