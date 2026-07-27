# Mechanistic Gene Attribution — 2026-07-27

Attributes the **Ridge/Variance k=85** downstream RFS classifier's decision back to the 40 genes of `all` for run `77d0103f`, by decomposing it through the 128-dim shared embedding space. Mechanistic (fixed-model) importance only — no leave-one-out retraining.

**Model provenance.** `77d0103f` is a refit (`scripts`/`hcc_multimodal.interpretability.refit_gene_encoder`, `--precompute-embeddings`) of `dc7e1d10`: the image encoder and its cached embeddings are reused **verbatim** (resection embedding cache is byte-identical), so the downstream head is identical to `dc7e1d10`'s by construction — its Setting-A grid and survival results (incl. head A1) carry over unchanged. Only the GeneEncoder is re-aligned to that frozen image space under a pinned, sorted gene order, giving an interpretable gene branch. The refit uses `--lam 0` (source `dc7e1d10` = 0.1) for 30 epochs: under a frozen image encoder the `per_modality` outcome reg on `z_img` is a no-op and the `z_gene` term is unopposed, so `lam=0` gives a pure image-alignment refit.

**Method.** (1) The grid head (`SimpleImputer(median) → StandardScaler → Variance(k=85) → Ridge`, α=100.0) is fit on the 54 resection patients' image embeddings vs `rfs_2year`, then collapsed into a single decision direction `β ∈ R¹²⁸` with `logit(z) = β·z + b`. (2) The GeneEncoder Jacobian `J[d,j] = ∂geneenc(g)_d/∂g_j` (at the cohort-mean gene vector) gives each gene's effect on each shared dim; `C[d,j] = β_d·J[d,j]` is its contribution to the decision axis. (3) Integrated Gradients of `s(g) = β·geneenc(g)` w.r.t. the gene input (200 midpoint steps, baseline = `zero`) aggregated over 60 patients as `mean|IG|` (importance) and `mean IG` (signed; **+ = pushes toward recurrence ≤ 2yr**).

> **Caveat.** Genes never enter the predictor at inference — they shape the image encoder only through the contrastive alignment at training time. This is an *alignment-mediated proxy*: the per-dimension decomposition of the decision axis on a re-aligned gene branch.

- Completeness check (max |Σ_j IG − (s(x)−s(baseline))|): **3.67e-02** = **2.41e-03** of the per-patient target range
- β reconstruction check (max |β·z+b − decision_function|): **4.44e-16**
- Resection 3-fold CV AUROC of the head: **0.744** (dc7e1d10 grid best cell = 0.744)
- Soramic transfer AUROC: **0.709** (dc7e1d10 best cell = 0.709); Lausanne: **0.436**

## Per-gene mechanistic importance (sorted by `mean|IG|`)

Two attribution targets, each decomposed to the genes by Integrated Gradients: the **downstream classification** decision axis (which genes move the RFS logit), and the **contrastive-learning alignment** score (which genes the gene encoder was actually trained to move). Each is reported under both a `zero` and a `cohort-mean` IG baseline.

### Downstream classification
**Method** 
1. Calc `logit(z) = β·z + b` from the downstream Ridge, where β is the rescaled coefficient.
2. Calc s(g) = β·gene_encoder(g) for each patients as a proxy of the 2-year RFS logits.
3. IG: for each patient, scale all the genes from the baseline to the real values in 200 steps. Use the gradients of the 200 steps to approximate the integral. Meaning: when we gradually increase the genes on a scale from 0 to their real values, the contribution of each genes on the (proxy)logit.

#### Baseline = 0
| Gene | mean\|IG\| | signed mean IG | pre-defined gene | rank |
|---|---:|---:|:---:|---:|
| SLC25A13 | 1.6568 | -1.5661 | ✓ | 1 |
| ACSM3 | 1.4851 | -1.4851 | ✓ | 2 |
| LACC1 | 1.4050 | +1.4050 | — | 3 |
| PON1 | 1.3818 | +1.1671 | ✓ | 4 |
| MYCBP2 | 1.0956 | +0.9146 | ✓ | 5 |
| ALS2 | 1.0873 | -0.8558 | ✓ | 6 |
| CFH | 1.0442 | +0.4452 | ✓ | 7 |
| SGSM1 | 1.0031 | +0.6155 | — | 8 |
| H19 | 0.9908 | +0.8004 | ✓ | 9 |
| USH1C | 0.9466 | +0.9306 | ✓ | 10 |
| M6PR | 0.9311 | -0.4539 | ✓ | 11 |
| SLC7A2 | 0.8655 | +0.6881 | ✓ | 12 |
| AP2B1 | 0.8348 | -0.1806 | ✓ | 13 |
| AC025580.2 | 0.8092 | -0.8092 | — | 14 |
| ABCB4 | 0.7938 | -0.0791 | ✓ | 15 |
| RALA | 0.7555 | +0.7110 | ✓ | 16 |
| AL445235.1 | 0.7181 | +0.5030 | — | 17 |
| HNRNPA1P9 | 0.6302 | +0.6042 | — | 18 |
| PDK4 | 0.6242 | +0.0162 | ✓ | 19 |
| REX1BD | 0.5970 | +0.2329 | ✓ | 20 |
| AC004241.5 | 0.5528 | -0.5330 | — | 21 |
| CAMK2N2 | 0.5516 | +0.4255 | — | 22 |
| ARF5 | 0.5169 | +0.4339 | ✓ | 23 |
| AC025198.1 | 0.4713 | +0.4713 | — | 24 |
| AOC1 | 0.4621 | -0.2575 | ✓ | 25 |
| AC093826.2 | 0.4118 | -0.3976 | — | 26 |
| AC093525.8 | 0.4062 | +0.0863 | — | 27 |
| CALCR | 0.3604 | -0.0992 | ✓ | 28 |
| AC138647.1 | 0.2944 | -0.2944 | — | 29 |
| RAD52 | 0.2816 | -0.0808 | ✓ | 30 |
| AL449283.1 | 0.2744 | +0.2610 | — | 31 |
| CYP51A1 | 0.2587 | -0.1633 | ✓ | 32 |
| MCUB | 0.2475 | -0.1672 | ✓ | 33 |
| OR52N5 | 0.2127 | +0.2101 | — | 34 |
| AC130366.1 | 0.1968 | +0.1394 | — | 35 |
| AC063947.2 | 0.1885 | +0.1766 | — | 36 |
| CSF2 | 0.1634 | -0.1634 | — | 37 |
| RBMXL3 | 0.1359 | -0.1359 | — | 38 |
| HIGD2B | 0.0911 | +0.0837 | — | 39 |
| ZMYND12 | 0.0897 | +0.0897 | — | 40 |

#### baseline = cohort-mean 

| Gene | mean\|IG\| | signed mean IG | pre-defined gene | rank |
|---|---:|---:|:---:|---:|
| LACC1 | 1.7951 | +0.1349 | — | 1 |
| ACSM3 | 1.6767 | -0.0820 | ✓ | 2 |
| AC025580.2 | 1.3457 | +0.1662 | — | 3 |
| PON1 | 1.1379 | +0.3638 | ✓ | 4 |
| USH1C | 1.1337 | -0.1230 | ✓ | 5 |
| HNRNPA1P9 | 0.9355 | +0.1522 | — | 6 |
| CAMK2N2 | 0.9114 | +0.0134 | — | 7 |
| PDK4 | 0.8770 | +0.1632 | ✓ | 8 |
| CSF2 | 0.7961 | +0.1455 | — | 9 |
| ALS2 | 0.7597 | -0.2242 | ✓ | 10 |
| AC025198.1 | 0.7471 | +0.1606 | — | 11 |
| SLC25A13 | 0.6823 | +0.0640 | ✓ | 12 |
| AOC1 | 0.6709 | +0.2787 | ✓ | 13 |
| SLC7A2 | 0.6647 | +0.1563 | ✓ | 14 |
| AC093826.2 | 0.6385 | -0.0168 | — | 15 |
| CALCR | 0.6328 | +0.1214 | ✓ | 16 |
| AC138647.1 | 0.5836 | -0.0050 | — | 17 |
| RALA | 0.5832 | +0.1750 | ✓ | 18 |
| M6PR | 0.5812 | -0.0379 | ✓ | 19 |
| AC004241.5 | 0.5757 | -0.1055 | — | 20 |
| AP2B1 | 0.5449 | +0.2621 | ✓ | 21 |
| ARF5 | 0.5164 | +0.0728 | ✓ | 22 |
| AC130366.1 | 0.4767 | -0.1156 | — | 23 |
| H19 | 0.4577 | +0.1950 | ✓ | 24 |
| OR52N5 | 0.4550 | +0.0595 | — | 25 |
| SGSM1 | 0.4461 | +0.0324 | — | 26 |
| CFH | 0.4349 | -0.2282 | ✓ | 27 |
| RAD52 | 0.4297 | -0.1390 | ✓ | 28 |
| REX1BD | 0.3986 | +0.1977 | ✓ | 29 |
| AL449283.1 | 0.3954 | +0.0777 | — | 30 |
| ABCB4 | 0.3531 | +0.0685 | ✓ | 31 |
| AL445235.1 | 0.3374 | -0.0198 | — | 32 |
| AC093525.8 | 0.3348 | -0.0668 | — | 33 |
| RBMXL3 | 0.3340 | +0.1467 | — | 34 |
| CYP51A1 | 0.3232 | -0.1526 | ✓ | 35 |
| MCUB | 0.3049 | +0.0659 | ✓ | 36 |
| AC063947.2 | 0.2978 | +0.1417 | — | 37 |
| HIGD2B | 0.2717 | +0.0116 | — | 38 |
| MYCBP2 | 0.2211 | -0.0147 | ✓ | 39 |
| ZMYND12 | 0.2176 | +0.1611 | — | 40 |

### Contrastive learning alignment
**Method**
1. Target is the cross-modal alignment score `F_i(g) = cos(z_img_i, gene_enc(g))` — patient *i*'s **frozen** image embedding vs the gene encoder output. 
2. IG of `F_i(g)` w.r.t. the gene input (200 midpoint steps), integrated only through the GeneEncoder, aggregated over the 60 patients as `mean|IG|` (importance) and `mean IG` (signed).
3. Sign convention: **signed IG > 0 ⇒ higher expression pulls the gene embedding *toward* the patient's own image**

#### Baseline = 0
| Gene | mean\|IG\| | signed mean IG | pre-defined gene | rank |
|---|---:|---:|:---:|---:|
| SLC25A13 | 0.1214 | +0.1094 | ✓ | 1 |
| ALS2 | 0.1030 | -0.0716 | ✓ | 2 |
| ABCB4 | 0.0928 | +0.0898 | ✓ | 3 |
| PON1 | 0.0890 | +0.0578 | ✓ | 4 |
| MYCBP2 | 0.0858 | +0.0630 | ✓ | 5 |
| CFH | 0.0850 | +0.0448 | ✓ | 6 |
| H19 | 0.0679 | +0.0148 | ✓ | 7 |
| M6PR | 0.0665 | -0.0614 | ✓ | 8 |
| AL445235.1 | 0.0603 | +0.0334 | — | 9 |
| CALCR | 0.0546 | +0.0537 | ✓ | 10 |
| SGSM1 | 0.0535 | +0.0309 | — | 11 |
| RALA | 0.0498 | +0.0361 | ✓ | 12 |
| REX1BD | 0.0498 | -0.0304 | ✓ | 13 |
| AP2B1 | 0.0486 | -0.0141 | ✓ | 14 |
| ACSM3 | 0.0477 | +0.0218 | ✓ | 15 |
| SLC7A2 | 0.0475 | +0.0211 | ✓ | 16 |
| PDK4 | 0.0439 | -0.0149 | ✓ | 17 |
| USH1C | 0.0436 | -0.0389 | ✓ | 18 |
| AC025580.2 | 0.0429 | -0.0412 | — | 19 |
| ARF5 | 0.0413 | -0.0277 | ✓ | 20 |
| AOC1 | 0.0343 | -0.0101 | ✓ | 21 |
| LACC1 | 0.0304 | +0.0153 | — | 22 |
| CSF2 | 0.0303 | -0.0303 | — | 23 |
| AC093525.8 | 0.0303 | -0.0098 | — | 24 |
| AC004241.5 | 0.0255 | +0.0104 | — | 25 |
| HNRNPA1P9 | 0.0249 | +0.0211 | — | 26 |
| RAD52 | 0.0241 | +0.0032 | ✓ | 27 |
| CYP51A1 | 0.0237 | -0.0062 | ✓ | 28 |
| AC063947.2 | 0.0234 | -0.0066 | — | 29 |
| AC025198.1 | 0.0225 | -0.0102 | — | 30 |
| ZMYND12 | 0.0214 | -0.0046 | — | 31 |
| AC093826.2 | 0.0199 | -0.0001 | — | 32 |
| MCUB | 0.0178 | +0.0007 | ✓ | 33 |
| AC130366.1 | 0.0152 | +0.0118 | — | 34 |
| AL449283.1 | 0.0140 | -0.0005 | — | 35 |
| CAMK2N2 | 0.0139 | -0.0017 | — | 36 |
| AC138647.1 | 0.0117 | +0.0001 | — | 37 |
| OR52N5 | 0.0107 | -0.0098 | — | 38 |
| HIGD2B | 0.0078 | -0.0078 | — | 39 |
| RBMXL3 | 0.0074 | +0.0004 | — | 40 |

#### baseline = cohort-mean
| Gene | mean\|IG\| | signed mean IG | pre-defined gene | rank |
|---|---:|---:|:---:|---:|
| CALCR | 0.0501 | -0.0005 | ✓ | 1 |
| ALS2 | 0.0439 | -0.0021 | ✓ | 2 |
| ABCB4 | 0.0429 | -0.0006 | ✓ | 3 |
| AC025580.2 | 0.0393 | -0.0050 | — | 4 |
| M6PR | 0.0376 | +0.0034 | ✓ | 5 |
| SLC7A2 | 0.0319 | -0.0005 | ✓ | 6 |
| PON1 | 0.0317 | -0.0008 | ✓ | 7 |
| REX1BD | 0.0313 | -0.0013 | ✓ | 8 |
| USH1C | 0.0310 | +0.0083 | ✓ | 9 |
| AC093525.8 | 0.0300 | +0.0048 | — | 10 |
| ACSM3 | 0.0231 | +0.0035 | ✓ | 11 |
| SLC25A13 | 0.0230 | +0.0079 | ✓ | 12 |
| PDK4 | 0.0223 | -0.0000 | ✓ | 13 |
| LACC1 | 0.0212 | -0.0074 | — | 14 |
| HNRNPA1P9 | 0.0211 | -0.0010 | — | 15 |
| CSF2 | 0.0199 | -0.0065 | — | 16 |
| AOC1 | 0.0179 | +0.0023 | ✓ | 17 |
| HIGD2B | 0.0175 | -0.0039 | — | 18 |
| AP2B1 | 0.0167 | +0.0011 | ✓ | 19 |
| AL449283.1 | 0.0167 | -0.0056 | — | 20 |
| AC093826.2 | 0.0166 | -0.0037 | — | 21 |
| AC063947.2 | 0.0164 | +0.0027 | — | 22 |
| AC004241.5 | 0.0163 | +0.0011 | — | 23 |
| ARF5 | 0.0162 | -0.0047 | ✓ | 24 |
| RALA | 0.0154 | -0.0020 | ✓ | 25 |
| ZMYND12 | 0.0148 | -0.0014 | — | 26 |
| RAD52 | 0.0143 | +0.0033 | ✓ | 27 |
| H19 | 0.0138 | -0.0068 | ✓ | 28 |
| CAMK2N2 | 0.0133 | -0.0040 | — | 29 |
| CYP51A1 | 0.0131 | -0.0013 | ✓ | 30 |
| AL445235.1 | 0.0130 | +0.0025 | — | 31 |
| AC130366.1 | 0.0127 | +0.0032 | — | 32 |
| MCUB | 0.0123 | -0.0034 | ✓ | 33 |
| RBMXL3 | 0.0122 | -0.0026 | — | 34 |
| AC025198.1 | 0.0122 | +0.0008 | — | 35 |
| CFH | 0.0121 | +0.0018 | ✓ | 36 |
| SGSM1 | 0.0114 | -0.0023 | — | 37 |
| OR52N5 | 0.0105 | -0.0074 | — | 38 |
| AC138647.1 | 0.0097 | +0.0041 | — | 39 |
| MYCBP2 | 0.0073 | -0.0005 | ✓ | 40 |

## Each gene's contribution on each dimension
C[d, j] = β_d · J[d, j], where J[d, j] is the Jacobian on the mean of gene vectors.
![Gene → decision-axis contribution heatmap](0727_mechanistic_interpretability_heatmap.png)


