# Mechanistic Gene Attribution — 2026-07-25

Attributes the **Ridge/Variance k=85** downstream RFS classifier's decision back to the 40 genes of `all` for run `92ae6a23`, by decomposing it through the 128-dim shared embedding space. Mechanistic (fixed-model) importance only — no leave-one-out retraining.

**Model provenance.** `92ae6a23` is a refit (`scripts`/`hcc_multimodal.interpretability.refit_gene_encoder`, `--precompute-embeddings`) of `dc7e1d10`: the image encoder and its cached embeddings are reused **verbatim** (resection embedding cache is byte-identical), so the downstream head is identical to `dc7e1d10`'s by construction — its Setting-A grid and survival results (incl. head A1) carry over unchanged. Only the GeneEncoder is re-aligned to that frozen image space under a pinned, sorted gene order, giving an interpretable gene branch.

**Method.** (1) The grid head (`SimpleImputer(median) → StandardScaler → Variance(k=85) → Ridge`, α=100.0) is fit on the 54 resection patients' image embeddings vs `rfs_2year`, then collapsed into a single decision direction `β ∈ R¹²⁸` with `logit(z) = β·z + b`. (2) The GeneEncoder Jacobian `J[d,j] = ∂geneenc(g)_d/∂g_j` (at the cohort-mean gene vector) gives each gene's effect on each shared dim; `C[d,j] = β_d·J[d,j]` is its contribution to the decision axis. (3) Integrated Gradients of `s(g) = β·geneenc(g)` w.r.t. the gene input (200 midpoint steps, baseline = `zero`) aggregated over 60 patients as `mean|IG|` (importance) and `mean IG` (signed; **+ = pushes toward recurrence ≤ 2yr**).

> **Caveat.** Genes never enter the predictor at inference — they shape the image encoder only through the contrastive alignment at training time. This is an *alignment-mediated proxy*: the per-dimension decomposition of the decision axis on a re-aligned gene branch.

- Completeness check (max |Σ_j IG − (s(x)−s(baseline))|): **4.66e-02** = **2.42e-03** of the per-patient target range
- β reconstruction check (max |β·z+b − decision_function|): **4.44e-16**
- Resection 3-fold CV AUROC of the head: **0.744** (dc7e1d10 grid best cell = 0.744)
- Soramic transfer AUROC: **0.709** (dc7e1d10 best cell = 0.709); Lausanne: **0.436**

## Per-gene mechanistic importance (sorted by `mean|IG|`)
**Method** 
1. Calc `logit(z) = β·z + b` from the downstream Ridge, where β is the rescaled coefficient.
2. Calc s(g) = β·gene_encoder(g) for each patients as a proxy of the 2-year RFS logits.
3. IG: for each patient, scale all the genes from 0 to the real values in 200 steps. Use the gradients of the 200 steps to approximate the integral. Meaning: when we gradually increase the genes on a scale from 0 to their real values, the contribution of each genes on the (proxy)logit.

The **pre-defined genes** column marks whether the gene is in the curated HCC gene panel (`get_hcc_genes()`: union of the `gene_sets/*.txt` sets, mapped to current symbols, intersected with the RNA-seq matrix → 2159 genes; see `hcc_multimodal/baselines/data.py`). ✓ = from the pre-defined panel, — = not. 22 of the 40 `all` genes come from the pre-defined panel; the remaining 18 (mostly `AC*`/`AL*` novel-transcript loci) do not.

| Gene | mean\|IG\| | signed mean IG | pre-defined gene | rank |
|---|---:|---:|:---:|---:|
| SLC25A13 | 2.2679 | -2.2679 | ✓ | 1 |
| LACC1 | 1.8017 | +1.8017 | — | 2 |
| ALS2 | 1.5598 | -1.3852 | ✓ | 3 |
| ACSM3 | 1.5556 | -1.5556 | ✓ | 4 |
| PON1 | 1.4742 | +1.2648 | ✓ | 5 |
| CFH | 1.4020 | +1.1685 | ✓ | 6 |
| MYCBP2 | 1.2906 | +1.1463 | ✓ | 7 |
| H19 | 1.1578 | +0.8657 | ✓ | 8 |
| SGSM1 | 1.1023 | +0.7519 | — | 9 |
| M6PR | 0.9972 | -0.3269 | ✓ | 10 |
| USH1C | 0.9907 | +0.9712 | ✓ | 11 |
| RALA | 0.9678 | +0.9457 | ✓ | 12 |
| AC025580.2 | 0.9524 | -0.9524 | — | 13 |
| SLC7A2 | 0.8935 | +0.7039 | ✓ | 14 |
| AP2B1 | 0.8748 | -0.0166 | ✓ | 15 |
| ABCB4 | 0.8316 | -0.1834 | ✓ | 16 |
| HNRNPA1P9 | 0.7127 | +0.7004 | — | 17 |
| PDK4 | 0.7063 | -0.0032 | ✓ | 18 |
| REX1BD | 0.6614 | +0.2793 | ✓ | 19 |
| ARF5 | 0.6397 | +0.5756 | ✓ | 20 |
| AL445235.1 | 0.6350 | +0.4057 | — | 21 |
| AC004241.5 | 0.5744 | -0.5171 | — | 22 |
| AC025198.1 | 0.5600 | +0.5600 | — | 23 |
| CAMK2N2 | 0.5484 | +0.4963 | — | 24 |
| AC093826.2 | 0.5395 | -0.5098 | — | 25 |
| AL449283.1 | 0.5005 | +0.5005 | — | 26 |
| AOC1 | 0.4902 | -0.2591 | ✓ | 27 |
| AC093525.8 | 0.4735 | +0.1301 | — | 28 |
| AC063947.2 | 0.3985 | +0.3955 | — | 29 |
| AC138647.1 | 0.3882 | -0.3882 | — | 30 |
| CSF2 | 0.3856 | -0.3856 | — | 31 |
| CYP51A1 | 0.3567 | -0.2950 | ✓ | 32 |
| CALCR | 0.3559 | -0.0439 | ✓ | 33 |
| AC130366.1 | 0.3506 | +0.3384 | — | 34 |
| MCUB | 0.3023 | -0.1469 | ✓ | 35 |
| RAD52 | 0.2869 | +0.0148 | ✓ | 36 |
| OR52N5 | 0.2466 | +0.2412 | — | 37 |
| ZMYND12 | 0.2138 | +0.2138 | — | 38 |
| HIGD2B | 0.1916 | +0.1916 | — | 39 |
| RBMXL3 | 0.1376 | -0.0943 | — | 40 |

## Each gene's contribution on each dimension
C[d, j] = β_d · J[d, j], where J[d, j] is the Jacobian on the mean of gene vectors.
![Gene → decision-axis contribution heatmap](0727_mechanistic_interpretability_heatmap.png)

## Appendix. Top downstream dimensions by |β| (with bootstrap stability)

Bootstrap = 500 stratified patient resamples. `sel. freq` = fraction of resamples where the selector keeps the dim. `top driver genes` = largest |C[d,j]|.

| dim | β | bootstrap mean±sd | sel. freq | top driver genes (signed C) |
|---|---:|---:|---:|---|
| 37 | +2.591 | +1.612±1.283 | 0.65 | CSF2 (-0.110), H19 (+0.094), HNRNPA1P9 (+0.089) |
| 74 | -2.216 | -1.453±1.674 | 0.70 | AC025198.1 (-0.087), AC138647.1 (+0.085), AL445235.1 (-0.085) |
| 11 | -1.707 | -0.936±0.876 | 0.61 | RBMXL3 (-0.064), RALA (+0.053), LACC1 (+0.047) |
| 98 | +1.593 | +1.023±0.827 | 0.74 | ACSM3 (-0.049), USH1C (+0.047), MCUB (+0.046) |
| 117 | -1.522 | -1.134±0.937 | 0.73 | AC138647.1 (-0.081), AL449283.1 (+0.068), ZMYND12 (+0.059) |
| 126 | +1.507 | +1.149±0.942 | 0.77 | AL445235.1 (-0.055), OR52N5 (+0.055), CSF2 (-0.054) |
| 35 | +1.437 | +0.929±0.852 | 0.62 | ABCB4 (+0.055), AC025198.1 (+0.050), AC063947.2 (-0.048) |
| 43 | -1.400 | -0.870±0.960 | 0.64 | LACC1 (+0.059), REX1BD (+0.046), AC093525.8 (+0.043) |
| 118 | +1.321 | +0.647±0.564 | 0.70 | CAMK2N2 (-0.042), AC138647.1 (+0.039), ZMYND12 (-0.038) |
| 80 | +1.250 | +0.963±0.942 | 0.70 | CSF2 (-0.037), AC025198.1 (+0.035), AC025580.2 (-0.034) |
| 40 | +1.087 | +0.726±0.871 | 0.65 | HIGD2B (+0.048), AC093525.8 (+0.038), AC063947.2 (+0.025) |
| 119 | -0.971 | -0.677±0.958 | 0.74 | AL449283.1 (+0.032), ZMYND12 (+0.029), RALA (+0.025) |
| 102 | +0.959 | +0.342±1.226 | 0.70 | AL445235.1 (-0.045), ALS2 (-0.043), REX1BD (-0.040) |
| 81 | -0.927 | -0.593±0.794 | 0.68 | RBMXL3 (+0.024), AOC1 (+0.024), ACSM3 (-0.021) |
| 9 | +0.899 | +0.578±0.576 | 0.58 | AL449283.1 (+0.056), AC138647.1 (-0.045), AC093826.2 (-0.040) |

**Notes**

- Stage-1 β says *which embedding dims the classifier uses*; stage-2 C and stage-3 IG say *which genes move the patient along those dims*. A gene ranks high only if it drives dims the classifier weights.
- Signed IG > 0 ⇒ higher expression pushes the patient toward 2-year recurrence along the classifier's decision axis.
