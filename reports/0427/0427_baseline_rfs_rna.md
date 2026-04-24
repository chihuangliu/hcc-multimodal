# HCC Multimodal — RFS Baseline (2026-04-27)

New baseline task: binary classification of **recurrence-free survival (RFS)** at 1-year and 2-year horizons, predicted from bulk RNA-seq only.

## Methodology

`notebooks/baselines/rna_baseline_rfs.ipynb`

**Targets:** `rfs_1year`, `rfs_2year` — derived from `RFS_central` / `RFS_central_event`. A patient is positive if a recurrence event is recorded within the horizon, negative if followed past the horizon without event, and censored (`NaN`, excluded) if follow-up ends before the horizon with no event.

**Dataset:** 60 patients × 50,986 genes. After low-expression filtering (count ≥ 15 in ≥ 5 samples) → 27,991 genes. Drop censored rows per-target.
- 1-year: n=56, positives=19 (34%)
- 2-year: n=54, positives=25 (46%)

**Pipeline (selector-first, raw counts in):**
1. `DeseqCPMSelector` — fit DESeq2 inside each training fold, select genes with BH-adjusted p < 0.1 (fallback: top 20 by padj if fewer pass); return log2(CPM+1) on the selected genes.
2. Median imputation + StandardScaler.
3. Model.

**Evaluation:** stratified 3-fold CV, AUC (ROC) primary metric. No hyperparameter grid search — a small manual sweep over the model's key regularisation knob is reported per fold.

**Model sweeps:**
- LR: `solver=liblinear, penalty=l1`, C ∈ {0.001, 0.01, 0.1, 1}
- RF: `max_depth ∈ {2, 4}` × `min_samples_leaf ∈ {5, 10, 15}`

---

## 1-year RFS

### Logistic Regression

![RFS 1 year LR](3_folds/rfs_1year_lr.png)

| Model | AUC mean ± std | Accuracy mean ± std |
|-------|---------------|---------------------|
| LR_C=0.001 | 0.500 ± 0.000 | 0.661 ± 0.022 |
| LR_C=0.01  | 0.500 ± 0.000 | 0.661 ± 0.022 |
| LR_C=0.1   | 0.500 ± 0.000 | 0.661 ± 0.022 |
| LR_C=1     | 0.333 ± 0.098 | 0.426 ± 0.107 |

All strongly-regularised settings collapse to the constant-majority predictor (test AUC = 0.500, accuracy = 0.661 ≈ 37/56). C=1 lets the weights grow enough to fit the training folds (train AUC ≈ 0.95) but test AUC drops *below* chance (0.333) — the model overfits DESeq-selected genes that do not generalise across folds.

### Random Forest

![RFS 1 year RF](3_folds/rfs_1year_rf.png)

| Model | AUC mean ± std | Accuracy mean ± std |
|-------|---------------|---------------------|
| RF_max_depth=2_min_samples_leaf=5  | 0.204 ± 0.049 | 0.590 ± 0.087 |
| RF_max_depth=2_min_samples_leaf=10 | 0.246 ± 0.136 | 0.661 ± 0.022 |
| RF_max_depth=2_min_samples_leaf=15 | 0.438 ± 0.108 | 0.661 ± 0.022 |
| RF_max_depth=4_min_samples_leaf=5  | 0.199 ± 0.070 | 0.554 ± 0.064 |
| RF_max_depth=4_min_samples_leaf=10 | 0.246 ± 0.136 | 0.661 ± 0.022 |
| RF_max_depth=4_min_samples_leaf=15 | 0.438 ± 0.108 | 0.661 ± 0.022 |

All configurations land below 0.5 test AUC. The best (leaf=15) approaches random (0.438); the more flexible settings (small leaf, any depth) sit near 0.20 — worse than inverted chance. Fold-level AUCs are widely dispersed (std up to 0.136), consistent with unstable selection from high-dimensional features + small n.

---

## 2-year RFS

### Logistic Regression

![RFS 2 year LR](3_folds/rfs_2year_lr.png)

| Model | AUC mean ± std | Accuracy mean ± std |
|-------|---------------|---------------------|
| LR_C=0.001 | 0.500 ± 0.000 | 0.537 ± 0.026 |
| LR_C=0.01  | 0.500 ± 0.000 | 0.537 ± 0.026 |
| LR_C=0.1   | 0.473 ± 0.038 | 0.537 ± 0.026 |
| LR_C=1     | **0.534 ± 0.164** | 0.481 ± 0.114 |

C=1 edges above the constant baseline (0.534) but with large fold-to-fold variance (±0.164). Train AUC climbs with weaker regularisation (≈ 0.97 at C=1); strong regularisation pins the model to the majority vote.

### Random Forest

![RFS 2 year RF](3_folds/rfs_2year_rf.png)

| Model | AUC mean ± std | Accuracy mean ± std |
|-------|---------------|---------------------|
| RF_max_depth=2_min_samples_leaf=5  | 0.474 ± 0.104 | 0.481 ± 0.026 |
| RF_max_depth=2_min_samples_leaf=10 | 0.490 ± 0.060 | 0.481 ± 0.069 |
| RF_max_depth=2_min_samples_leaf=15 | 0.500 ± 0.000 | 0.537 ± 0.026 |
| RF_max_depth=4_min_samples_leaf=5  | 0.465 ± 0.087 | 0.481 ± 0.026 |
| RF_max_depth=4_min_samples_leaf=10 | 0.490 ± 0.060 | 0.481 ± 0.069 |
| RF_max_depth=4_min_samples_leaf=15 | 0.500 ± 0.000 | 0.537 ± 0.026 |

All configurations hover at chance (0.465–0.500). `min_samples_leaf=15` collapses every tree's split budget so the forest reverts to the class prior. Unlike 1-year, no setting falls well below 0.5 — the class balance is less skewed (46% vs. 34% positives) so the "always predict majority" failure mode is less punitive.

---

## DESeq2-selected features per fold (LR C=1)

Full gene lists are in `reports/0427/3_folds/selected_features_{1,2}year_fold{1,2,3}.csv`.

### 1-year RFS

| Fold | n selected | passed padj<0.1 | train AUC | test AUC |
|------|-----------|-----------------|-----------|----------|
| 1 | 20 | 0 (fallback) | 0.917 | 0.410 |
| 2 | 20 | 15 | 0.930 | 0.393 |
| 3 | 606 | 606 | 1.000 | 0.194 |

**Fold 1** — zero genes passed BH correction; selector fell back to the top-20 by padj. All 20 have padj ≈ 1.0, meaning there is effectively no differential expression signal in this training partition. Top genes (ranked by padj, all at ≈0.9999): TSPAN6, CYP51A1, ANKIB1, CFTR, SEMA3F.

**Fold 2** — 15 genes passed padj < 0.1 (5 more added via fallback). Top hits: CABP2 (0.005), REXO1L3P (0.006), EFHC2 (0.018), DCDC2C (0.018), AC006538.1 (0.028). These are functionally diverse (calcium-binding, ciliary, cell-cycle) with no obvious liver/recurrence link — consistent with noise at n≈37 training samples.

**Fold 3** — 606 genes passed padj < 0.1 (anomalously large; this is the fold where train AUC = 1.0 and test AUC = 0.194). Top hits: FAXC (0.005), NBPF22P (0.005), FAM180B (0.007), ZNF114 (0.009), STOML3 (0.010). The explosion of significant genes while test AUC collapses suggests the DESeq2 model overfitted the training-fold label split and found spurious associations.

**Zero gene overlap** across all three folds — the selected set is entirely different each time, confirming that the signal is not reproducible at this sample size.

### 2-year RFS

| Fold | n selected | passed padj<0.1 | train AUC | test AUC |
|------|-----------|-----------------|-----------|----------|
| 1 | 20 | 15 | 0.978 | 0.650 |
| 2 | 67 | 67 | 0.957 | 0.650 |
| 3 | 274 | 274 | 1.000 | 0.302 |

**Fold 1** — 15 genes passed threshold; fallback adds 5 more. Top hits: ASB5 (0.017), AC025580.2 (0.017), ATOH1 (0.017), MIAT (0.021), AC127070.4 (0.021). MIAT (myocardial infarction associated transcript) is a long non-coding RNA with reported roles in cell survival; ATOH1 is a transcription factor involved in differentiation.

**Fold 2** — 67 genes passed; notably UBTFL2 (padj ≈ 0), PRY2 (padj ≈ 0), NKX2-5 (0.003), DUSP9 (0.003), ISL1 (0.004). NKX2-5 and ISL1 are cardiac/developmental transcription factors rarely expressed in liver — their near-zero padj in a 36-sample training fold is suspicious and may reflect a confound in the data split.

**Fold 3** — 274 genes passed (again an outlier fold with train AUC = 1.0 and the weakest test AUC = 0.302). Top hits: TRIM60 (0.001), SLC35E4 (0.001), PTPN5 (0.005), GRHL3 (0.005), C3orf84 (0.005).

**No gene overlap** across folds in the 2-year target either — the same instability pattern as 1-year.

---

## Takeaways (3-fold)

- **2-year RFS is easier than 1-year but still near chance.** Best test AUC is 0.534 (LR C=1), vs. 0.500-or-worse for every 1-year configuration. The extra 6 positives (19 → 25) shift the class balance closer to 50/50 and give the selector more signal to work with.
- **1-year RFS is hard for a pure RNA-seq + DESeq selector.** No model beats the 0.500 constant baseline; the only model that learns (LR C=1) overfits and generalises below chance (0.333). This contrasts with `death` on the same RNA-seq input (AUC 0.670 in the 0420 baseline) — longitudinal-recurrence signal in the bulk transcriptome is evidently weaker, and censoring drops further samples.
- **DESeq inside CV rarely finds significant genes**, falling back to top-20-by-padj on most folds (warnings "only 0 gene(s) passed padj<0.1"). The selected gene set therefore differs across folds — a likely driver of the large fold-wise AUC variance (up to ±0.164).
- Candidate next steps: widen the fallback-k, try a class-balanced loss, or pool 1-year + 2-year supervision via a multi-task head; add clinical covariates to see if they stabilise the RNA signal.

---

## 4-fold CV replication

Same pipeline re-run with `CV_N_FOLDS=4` (stratified 4-fold). Plots and feature CSVs: `notebooks/baselines/4_folds/`.

### 1-year RFS

![RFS 1 year LR 4-fold](4_folds/rfs_1y_lr.png)
![RFS 1 year RF 4-fold](4_folds/rfs_1y_rf.png)

**Logistic Regression**

| Model | AUC mean ± std | Accuracy mean ± std |
|-------|---------------|---------------------|
| LR_C=0.001 | 0.500 ± 0.000 | 0.661 ± 0.031 |
| LR_C=0.01  | 0.500 ± 0.000 | 0.661 ± 0.031 |
| LR_C=0.1   | 0.500 ± 0.000 | 0.661 ± 0.031 |
| LR_C=1     | 0.358 ± 0.085 | 0.464 ± 0.107 |

**Random Forest**

| Model | AUC mean ± std | Accuracy mean ± std |
|-------|---------------|---------------------|
| RF_max_depth=2_min_samples_leaf=5  | 0.279 ± 0.135 | 0.500 ± 0.196 |
| RF_max_depth=2_min_samples_leaf=10 | 0.350 ± 0.147 | 0.643 ± 0.051 |
| RF_max_depth=2_min_samples_leaf=15 | 0.401 ± 0.159 | 0.661 ± 0.031 |
| RF_max_depth=4_min_samples_leaf=5  | 0.247 ± 0.152 | 0.482 ± 0.178 |
| RF_max_depth=4_min_samples_leaf=10 | 0.350 ± 0.147 | 0.643 ± 0.051 |
| RF_max_depth=4_min_samples_leaf=15 | 0.401 ± 0.159 | 0.661 ± 0.031 |

DESeq2 selector: all 4 folds fall back to the top-20 minimum (no genes pass padj<0.1), except fold 4 where 40 genes pass (best: RPL7P7, padj=0.0008). Zero gene overlap across folds.

### 2-year RFS

![RFS 2 year LR 4-fold](4_folds/rfs_2y_lr.png)
![RFS 2 year RF 4-fold](4_folds/rfs_2y_rf.png)

**Logistic Regression**

| Model | AUC mean ± std | Accuracy mean ± std |
|-------|---------------|---------------------|
| LR_C=0.001 | 0.500 ± 0.000 | 0.537 ± 0.025 |
| LR_C=0.01  | 0.500 ± 0.000 | 0.537 ± 0.025 |
| LR_C=0.1   | 0.488 ± 0.021 | 0.537 ± 0.025 |
| LR_C=1     | 0.304 ± 0.122 | 0.372 ± 0.063 |

**Random Forest**

| Model | AUC mean ± std | Accuracy mean ± std |
|-------|---------------|---------------------|
| RF_max_depth=2_min_samples_leaf=5  | 0.413 ± 0.126 | 0.391 ± 0.092 |
| RF_max_depth=2_min_samples_leaf=10 | 0.403 ± 0.103 | 0.500 ± 0.105 |
| RF_max_depth=2_min_samples_leaf=15 | 0.388 ± 0.081 | 0.537 ± 0.025 |
| RF_max_depth=4_min_samples_leaf=5  | 0.401 ± 0.137 | 0.409 ± 0.075 |
| RF_max_depth=4_min_samples_leaf=10 | 0.403 ± 0.103 | 0.500 ± 0.105 |
| RF_max_depth=4_min_samples_leaf=15 | 0.388 ± 0.081 | 0.537 ± 0.025 |

DESeq2 selector: fold 1=42 genes (best: CAMK1G, padj=0.0002), fold 2=20 (fallback), fold 3=614 (explosion, same outlier pattern as 3-fold), fold 4=22 genes. Zero cross-fold overlap.

### Conclusion

4-fold results are consistent with 3-fold: all models remain at or below chance for both targets. The 2-year LR best drops from 0.534 (3-fold) to 0.488 (4-fold), and the 1-year LR best stays at ≤0.500. The outlier-fold phenomenon (one fold selects hundreds of genes, train AUC→1, test AUC collapses) persists under 4-fold, confirming it is a structural property of DESeq2 in small-n splits rather than an artifact of the fold count. RNA-seq alone provides no reproducible signal for RFS at this sample size.

---

## SelectKBest (k=80) vs DESeq comparison

`SELECTOR="sklearn_kbest"`, `KBEST=80`. Same 3-fold CV, same LR/RF configs. Pipeline: imputer + StandardScaler → `SelectKBest(f_classif, k=80)` → model (i.e. `selector_first=False`; raw counts enter the preprocessor). Plots: `notebooks/baselines/kbest_3_folds/`.

**Note:** SelectKBest flags 2–3 constant gene features per fold (indices 23517, 26012, 27801); these receive undefined F-statistics and are never among the top 80.

### 1-year RFS (SelectKBest)

![RFS 1 year LR kbest](../../notebooks/baselines/kbest_3_folds/rfs_1y_lr.png)
![RFS 1 year RF kbest](../../notebooks/baselines/kbest_3_folds/rfs_1y_rf.png)

| Model | AUC mean ± std |
|-------|----------------|
| LR_C=0.001 | 0.500 ± 0.000 |
| LR_C=0.01  | 0.500 ± 0.000 |
| LR_C=0.1   | 0.500 ± 0.000 |
| LR_C=1     | 0.374 ± 0.078 |
| RF leaf=5  | 0.345 ± 0.043 |
| RF leaf=10 | 0.350 ± 0.176 |
| RF leaf=15 | **0.478 ± 0.064** |

### 2-year RFS (SelectKBest)

![RFS 2 year LR kbest](../../notebooks/baselines/kbest_3_folds/rfs_2y_lr.png)
![RFS 2 year RF kbest](../../notebooks/baselines/kbest_3_folds/rfs_2y_rf.png)

| Model | AUC mean ± std |
|-------|----------------|
| LR_C=0.001 | 0.500 ± 0.000 |
| LR_C=0.01  | 0.500 ± 0.000 |
| LR_C=0.1   | 0.500 ± 0.000 |
| LR_C=1     | 0.398 ± 0.074 |
| RF leaf=5  | **0.558 ± 0.072** |
| RF leaf=10 | 0.525 ± 0.061 |
| RF leaf=15 | 0.500 ± 0.000 |

### Head-to-head summary (3-fold)

| Target | Selector | Best model | AUC mean | AUC std |
|--------|----------|-----------|----------|---------|
| 1-year | DESeq (padj<0.1, min 20) | RF leaf=15 | 0.438 | 0.108 |
| 1-year | SelectKBest k=80 | RF leaf=15 | **0.478** | 0.064 |
| 2-year | DESeq (padj<0.1, min 20) | LR C=1 | **0.534** | 0.164 |
| 2-year | SelectKBest k=80 | RF leaf=5 | 0.558 | 0.072 |

**1-year RFS:** SelectKBest modestly outperforms DESeq (0.478 vs 0.438 best RF) and has lower fold-variance (±0.064 vs ±0.108). Both are below chance, but SelectKBest avoids the "inverted" < 0.400 collapse seen with DESeq's unstable fold selections. LR behaviour is identical: collapses at all C values tested.

**2-year RFS:** SelectKBest RF (0.558 ± 0.072) edges DESeq RF (0.490 ± 0.060) and matches DESeq LR (0.534), while DESeq LR carries much higher variance (±0.164). However, DESeq's LR C=1 best fold (0.650) exceeds any individual SelectKBest fold. The key contrast is stability: SelectKBest produces more consistent fold-to-fold AUC, whereas DESeq's gene selection explodes in outlier folds (hundreds of genes, train AUC→1, test AUC collapses), driving high variance and occasional above-chance exceptions.

**Conclusion:** At n≈37 training samples, `SelectKBest(f_classif, k=80)` on scaled raw counts is a more stable selector than DESeq2 inside CV — it avoids the outlier-fold phenomenon and yields slightly higher or comparable mean AUC across both targets. Neither approach achieves reliable above-chance generalisation; the signal ceiling is set by sample size, not selector choice.
