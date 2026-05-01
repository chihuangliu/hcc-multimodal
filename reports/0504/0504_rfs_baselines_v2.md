
## Table of Contents

- [1. Task](#1-task)
- [2. Key findings](#2-key-findings)
- [3. RNA-seq vs. radiomics](#3-rna-seq-vs-radiomics)
  - [Feature selection in cross-validation](#feature-selection-in-cross-validation)
  - [Feature selection before cross-validation](#feature-selection-before-cross-validation)
  - [Plot AUC in each fold](#plot-auc-in-each-fold)
- [4. Predefined HCC gene set](#4-predefined-hcc-gene-set)
  - [HCC Gene set description](#hcc-gene-set-description)
  - [Predefined HCC gene set CV results](#predefined-hcc-gene-set-cv-results)
- [5. Genes feature selection analysis](#5-genes-feature-selection-analysis)
  - [Pre CV selected features](#pre-cv-selected-features)
  - [Preselected features vs predefined HCC gene set](#preselected-features-vs-predefined-hcc-gene-set)
  - [Non-zero LR coefficient features per fold](#non-zero-lr-coefficient-features-per-fold-feature-selection-before-cross-validation)
- [6. Ensemble (radiomics + RNA-seq)](#6-ensemble-radiomics--rna-seq)

# 1. Task
1. Binary classification of recurrence-free survival (RFS) at 1-year and 2-year horizons using two modalities: RNA-seq (DESeq2 with `padj < 0.05`, 3-fold stratified CV) and arterial-phase CT radiomics (SelectKBest F-score, k=100, 3-fold CV). Experiments span two selector placements — in-CV vs before-CV  
2.  Experiment with a predefined HCC gene set.
3. Analyze the selected gene features.

# 2. Key findings
- 1 year rfs: Arterial radiomics outperform RNA-seq, not matter the selector placement and the model. [See 3.RNA-seq vs. radiomics](#3-rna-seq-vs-radiomics)
- 2 year rfs: RNA-seq can outperform Arterial radiomics but is model dependent: if feature selection in-CV, RNA-seq with Random Forest is the best (AUC=0.581); if feature selection before CV, RNA-seq with Logistic Regression is the best (AUC=0.864). [See 3.RNA-seq vs. radiomics](#3-rna-seq-vs-radiomics)
- Predefined HCC gene set didn't work well. [See 4.Predefined HCC gene set - CV results](#predefined-hcc-gene-set-cv-results)
- The pre-CV selected genes on our data set shows little overlap with predefined HCC gene set. [See Preselected features vs predefined HCC gene set](#preselected-features-vs-predefined-hcc-gene-set)
- 8~13 pre-CV selected genes have non-zero LR coefficient. 1y-RFS CV have 3 common ones in all 3 folds, and 2y-RFS have 7 common ones. Statistical significance does not mean a gene can have a non-zero coefficient.[See Non-zero LR coefficient features per fold](#non-zero-lr-coefficient-features-per-fold-feature-selection-before-cross-validation)

# 3. RNA-seq vs. radiomics
## Feature selection in cross-validation

Source: `reports/0427`.

| Modality | Selector | 1y LR | 1y RF | 2y LR | 2y RF |
|----------|----------|-------|-------|-------|-------|
| RNA-seq | DESeq2 (`padj < 0.1`, min 20 features)| 0.333 | 0.438 | 0.517 | **0.581** |
| Arterial radiomics | SelectKBest F-score (k=100) | 0.654 | **0.665** | 0.537 | 0.570 |

## Feature Selection before cross-validation

Source: `reports/0504`.

| Modality | Selector | 1y LR | 1y RF | 2y LR | 2y RF |
|----------|----------|-------|-------|-------|-------|
| Arterial radiomics | SelectKBest F-score (k=100) | **0.819** | 0.793 | 0.798 | 0.761 |
| RNA-seq | DESeq2 (`padj < 0.05`, min 20 features) | 0.686 | 0.601 | **0.864** | 0.761 |

## Plot AUC in each fold

Circles = per-fold AUC, diamonds = mean.

### Feature selection in cross-validation

#### RNA-seq — DESeq2

| | 1-year RFS | 2-year RFS |
|---|---|---|
| **LR** | ![LR 1y](../0427/3_folds/rfs_1year_lr.png) | ![LR 2y](../0427/3_folds/rfs_2year_lr.png) |
| **RF** | ![RF 1y](../0427/3_folds/rfs_1year_rf.png) | ![RF 2y](../0427/3_folds/rfs_2year_rf.png) |

#### Arterial radiomics — SelectKBest (F-score, k=100)

| | 1-year RFS | 2-year RFS |
|---|---|---|
| **LR** | ![LR 1y](../../notebooks/baselines/arterial_rfs/rfs_1y_lr.png) | ![LR 2y](../../notebooks/baselines/arterial_rfs/rfs_2y_lr.png) |
| **RF** | ![RF 1y](../../notebooks/baselines/arterial_rfs/rfs_1y_rf.png) | ![RF 2y](../../notebooks/baselines/arterial_rfs/rfs_2y_rf.png) |

### Feature selection before cross-validation
#### Arterial radiomics — SelectKBest (F-score, k=100)

| | 1-year RFS | 2-year RFS |
|---|---|---|
| **LR** | ![LR 1y](kbest_f100_before_cv/rfs_1y_lr.png) | ![LR 2y](kbest_f100_before_cv/rfs_2y_lr.png) |
| **RF** | ![RF 1y](kbest_f100_before_cv/rfs_1y_rf.png) | ![RF 2y](kbest_f100_before_cv/rfs_2y_rf.png) |

#### RNA-seq — DESeq2, all genes

| | 1-year RFS | 2-year RFS |
|---|---|---|
| **LR** | ![LR 1y](deseq_p0.05_before_cv_all_genes/rfs_1y_lr.png) | ![LR 2y](deseq_p0.05_before_cv_all_genes/rfs_2y_lr.png) |
| **RF** | ![RF 1y](deseq_p0.05_before_cv_all_genes/rfs_1y_rf.png) | ![RF 2y](deseq_p0.05_before_cv_all_genes/rfs_2y_rf.png) |


# 4. Predefined HCC gene set
## HCC Gene set description 
- Total number: 2,175 (unique gene symbols across 12 HCC gene sets in `data/RNA_seq/gene_sets_combined/`)
- Directly match to RNA-seq data set (`/data/RNA_seq/Matrix_output_radiology_only.csv`): 2,160
- 1-on-1 mapped to RNA-seq data set: 31

  | Gene set symbol | Matrix symbol | Rationale |
  |----------------|---------------|-----------|
  | AARS1 | AARS | HGNC approved symbol update |
  | BPNT2 | IMPAD1 | HGNC approved symbol update |
  | CCN1 | CYR61 | HGNC approved symbol update |
  | CCN2 | CTGF | HGNC approved symbol update |
  | CILK1 | ICK | HGNC approved symbol update |
  | DYNC2I2 | WDR60 | HGNC approved symbol update |
  | EPRS1 | EPRS | HGNC approved symbol update |
  | G6PC1 | G6PC | HGNC approved symbol update |
  | GBA1 | GBA | HGNC approved symbol update |
  | H1-0 | H1F0 | HGNC approved symbol update |
  | H1-6 | HIST1H1E | HGNC approved symbol update |
  | H2BC7 | HIST1H2BG | HGNC approved symbol update |
  | H2BC15 | HIST1H2BN | HGNC approved symbol update |
  | H2BC21 | HIST2H2BF | HGNC approved symbol update |
  | H4C1 | HIST1H4A | HGNC approved symbol update |
  | H4C3 | HIST1H4C | HGNC approved symbol update |
  | H4C14 | HIST2H4B | HGNC approved symbol update |
  | HJV | HFE2 | HGNC approved symbol update |
  | IARS1 | IARS | HGNC approved symbol update |
  | ILRUN | C6orf106 | HGNC approved symbol update |
  | ITPRID2 | PPIP5K2 | HGNC approved symbol update |
  | MACROH2A2 | H2AFY2 | HGNC approved symbol update |
  | MMUT | MUT | HGNC approved symbol update |
  | NARS1 | NARS | HGNC approved symbol update |
  | NHERF1 | SLC9A3R1 | HGNC approved symbol update |
  | NHERF2 | SLC9A3R2 | HGNC approved symbol update |
  | NIBAN1 | FAM129A | HGNC approved symbol update |
  | NTAQ1 | NTAN1 | HGNC approved symbol update |
  | PHB1 | PHB | HGNC approved symbol update |
  | PLAAT3 | PLA2G16 | HGNC approved symbol update |
  | RSC1A1 | SLC7A9 | HGNC approved symbol update |

- Cannot match: 15 — symbol absent from RNA-seq matrix and no clear matches:

  | Gene | Reason no distinct mapping can be defined |
  |------|------------------------------------------|
  | CIDEB | No known prior symbol; gene entirely absent from quantification reference (annotation version gap) |
  | FLJ30679 | Provisional FLJ-series EST name; potential HGNC successor CFAP97D1 is in the matrix, but FLJ names may map to multiple genomic loci — confirmed 1-to-1 correspondence requires HGNC validation |
  | GABARAPL3 | Annotated as a pseudogene (HGNC:33517); no protein-coding alias exists for an unambiguous mapping |
  | ID2B | Pseudogene of ID2 (which is in the matrix as a distinct entry); mapping pseudogene to its parent would conflate different genomic loci |
  | LINC02693 | lncRNA with prior chromosomal alias C3orf67 in the matrix, but LINC02693 coordinate boundaries differ from the older ORF definition across annotation versions |
  | MARCHF3 | Updated MARCHF-prefix symbol (was MARCH3); old symbol MARCH3 is also absent from the matrix — gene missing from quantification reference entirely |
  | METTL25B | Paralog of METTL25 (which is in the matrix as a distinct entry); METTL25B and METTL25 are different genes — mapping would conflate two distinct methyltransferases |
  | MTARC2 | Prior symbols MOSC2 and MARC2 also absent from the matrix — gene missing from quantification reference entirely |
  | PABIR2 | Two potential matrix aliases found (APPBP2, TNRC6C) — no distinct 1-to-1 mapping can be assigned |
  | PIGAP1 | Annotated as a pseudogene; no protein-coding alias present in the matrix |
  | POU2AF3 | Recently characterized gene; prior chromosomal alias C11orf53 is in the matrix but may encompass a broader locus not co-extensive with POU2AF3 |
  | PTGR3 | Prior alias ZADH2 is in the matrix, but ZADH2 was reclassified as PTGR3 only in recent HGNC releases — unambiguous correspondence requires HGNC validation |
  | SEPTIN4 | Updated SEPTIN-prefix symbol (was SEPT4); old symbol SEPT4 also absent from the matrix — gene missing from quantification reference entirely |
  | SEPTIN7 | Updated SEPTIN-prefix symbol (was SEPT7); old symbol SEPT7 also absent from the matrix — gene missing from quantification reference entirely |
  | UTP25 | Prior symbol KIAA1041 also absent from the matrix — gene missing from quantification reference entirely |

## Predefined HCC gene set CV results

Best mean test AUC for C2 (before-CV, predefined genes) and C3 (in-CV, predefined genes). All runs use `padj < 0.05`; results are identical at `padj < 0.1` (no gene passes BH in either case — fallback-dominated throughout).

| Config | Selector placement | 1y LR | 1y RF | 2y LR | 2y RF |
|--------|--------------------|-------|-------|-------|-------|
| C2 — before_cv, predefined | before CV | 0.500 | 0.570 | 0.500 | 0.505 |
| C3 — in_cv, predefined | inside CV | 0.500 | 0.500 | 0.500 | 0.508 |

# 5. Genes feature selection analysis
## Pre CV selected features 
padj threshold was set as 0.05, but there were too few below it, so 20 features with the min padj were selected.
### 1-year RFS

| Feature | padj |
|---------|------|
| AL138889.1 | 5.84e-04 *|
| AC135731.1 | 5.84e-04 *|
| AC004889.1 | 3.33e-02 *|
| AL160272.1 | 5.61e-02 |
| INAFM2 | 5.61e-02 |
| AC092068.2 | 6.24e-02 |
| LINC00514 | 6.24e-02 |
| AL031594.1 | 1.06e-01 |
| AL117328.2 | 1.19e-01 |
| AC005696.4 | 1.65e-01 |
| CCDC26 | 1.65e-01 |
| AC025580.2 | 2.11e-01 |
| EGFL8 | 2.11e-01 |
| AC006538.1 | 2.11e-01 |
| AC127070.4 | 2.53e-01 |
| AL353726.2 | 2.86e-01 |
| PCSK6-AS1 | 2.92e-01 |
| AC016355.1 | 3.13e-01 |
| AC008395.1 | 3.29e-01 |
| AL008638.3 | 3.29e-01 |

### 2-year RFS

| Feature | padj |
|---------|------|
| CAMK2N2 | 8.94e-03 *|
| AC093826.2 | 3.18e-02 *|
| LACC1 | 4.58e-02 *|
| AC093525.8 | 1.98e-01 |
| CSF2 | 1.98e-01 |
| AC022098.1 | 1.98e-01 |
| AC025198.1 | 1.98e-01 |
| AC025580.2 | 1.98e-01 |
| AC004241.5 | 1.98e-01 |
| AL445235.1 | 1.98e-01 |
| OR52N5 | 2.15e-01 |
| HNRNPA1P9 | 2.88e-01 |
| AL449283.1 | 3.03e-01 |
| LINC02241 | 5.18e-01 |
| AC138647.1 | 5.18e-01 |
| HIGD2B | 5.18e-01 |
| SGSM1 | 5.18e-01 |
| H19 | 5.18e-01 |
| CCR12P | 5.18e-01 |
| ZMYND12 | 5.18e-01 |

## Preselected features vs predefined HCC gene set
  <img src="venns/c1_vs_hcc.png" width="600">

## Non-zero LR coefficient features per fold (Feature Selection before cross-validation)
  <img src="venns/c1_lr_nonzero.png" width="600">

### 1-year RFS — non-zero features per fold

| Feature | padj | HCC set | Fold 1 coef | Fold 2 coef | Fold 3 coef |
|---------|------|---------|-------------|-------------|-------------|
| AC135731.1 | 5.84e-04 * | No | +0.240 | +0.537 | −0.112 |
| AL138889.1 | 5.84e-04 * | No | +0.222 | — | — |
| AC004889.1 | 3.33e-02 * | No | −0.173 | — | −0.284 |
| LINC00514 | 6.24e-02 | No | −0.260 | −0.225 | −0.209 |
| AC092068.2 | 6.24e-02 | No | — | −0.202 | −0.592 |
| AL160272.1 | 5.61e-02 | No | — | — | −0.178 |
| INAFM2 | 5.61e-02 | No | — | −0.512 | — |
| EGFL8 | 2.11e-01 | No | — | −0.520 | — |
| AC025580.2 | 2.11e-01 | No | −0.461 | — | −0.665 |
| AC005696.4 | 1.65e-01 | No | −0.081 | −0.082 | −0.615 |
| AC006538.1 | 2.11e-01 | No | −0.238 | — | — |
| AC016355.1 | 3.13e-01 | No | −0.366 | — | — |
| AC127070.4 | 2.53e-01 | No | −0.336 | — | −0.340 |
| PCSK6-AS1 | 2.92e-01 | No | −0.322 | −0.431 | — |
| AC008395.1 | 3.29e-01 | No | — | −0.033 | −0.396 |

Stable across all 3 folds: `AC135731.1` (padj < 0.05), `LINC00514`, `AC005696.4`. No feature is in the HCC predefined set.

### 2-year RFS — non-zero features per fold

| Feature | padj | HCC set | Fold 1 coef | Fold 2 coef | Fold 3 coef |
|---------|------|---------|-------------|-------------|-------------|
| LACC1 | 4.58e-02 * | No | +1.480 | +0.130 | +0.456 |
| AC093826.2 | 3.18e-02 * | No | −0.605 | −0.888 | −0.327 |
| CAMK2N2 | 8.94e-03 * | No | — | +0.187 | — |
| AC025580.2 | 1.98e-01 | No | −0.773 | −0.493 | −0.619 |
| AL449283.1 | 3.03e-01 | No | +0.636 | +1.018 | +1.021 |
| SGSM1 | 5.18e-01 | No | +0.687 | +0.804 | +0.734 |
| AL445235.1 | 1.98e-01 | No | −0.322 | +0.018 | −0.867 |
| H19 | 5.18e-01 | **Yes** | −0.227 | −0.667 | −0.284 |
| ZMYND12 | 5.18e-01 | No | — | +0.969 | +0.507 |
| AC004241.5 | 1.98e-01 | No | −0.263 | −0.266 | — |
| CCR12P | 5.18e-01 | No | — | +0.590 | +0.247 |
| AC093525.8 | 1.98e-01 | No | — | −0.426 | −0.294 |
| AC138647.1 | 5.18e-01 | No | −0.247 | — | — |
| AC022098.1 | 1.98e-01 | No | — | −0.166 | — |
| LINC02241 | 5.18e-01 | No | — | — | +0.221 |
| HNRNPA1P9 | 2.88e-01 | No | — | — | +0.198 |
| HIGD2B | 5.18e-01 | No | — | — | −0.062 |

Stable across all 3 folds: `LACC1`, `AC093826.2`, `AC025580.2`, `AL449283.1`, `SGSM1`, `AL445235.1`, `H19`. only `LACC1` and `AC093826.2` pass padj < 0.05. `H19` (the is in predefined HCC set) is non-zero in all folds despite padj = 0.518.

# 6. Ensemble (radiomics + RNA-seq)

Average of LR C=1 predicted probabilities from arterial radiomics and RNA-seq (both before-CV, 2-year RFS). Source: `notebooks/baselines/ensemble_baseline_rfs.ipynb`.

| Fold | Ensemble | Radiomics | RNA-seq |
|------|----------|-----------|---------|
| 1 | 0.912 | 0.825 | 0.913 |
| 2 | 0.926 | 0.901 | 0.840 |
| 3 | 0.815 | 0.667 | 0.790 |
| **mean** | **0.884** | 0.798 | 0.847 |

The ensemble outperforms both single modalities in mean AUC. Improvement is most pronounced in fold 3, where both individual modalities are weakest.

<img src="ensemble_2y_lr/ensemble_2y_lr_auc.png" width="500">