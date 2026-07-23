# Embedding Distribution Drift — KS-Test Analysis
**Date:** 2026-06-15

---

## 1. Setup

For each of the 17 contrastive models, patient-level embeddings (128-dim, mean-pooled over sagittal slices) were loaded from the cached parquet files for three cohorts:

- **Resection** — training cohort (n=60 patients)
- **Soramic** — ablation test cohort (n=100 patients)
- **Lausanne** — external test cohort (n=68 patients)

A per-dimension Kolmogorov–Smirnov (KS) test was run for each cohort pair. Three summary statistics are reported per model per comparison:

| Statistic | Definition |
|-----------|-----------|
| `median_d` | Median KS D-statistic across 128 dimensions |
| `mean_d` | Mean KS D-statistic across 128 dimensions |
| `frac_sig` | Fraction of dimensions with p < 0.05 |

Script: `hcc_multimodal/eval/embedding_drift.py`  
Output: `results/eval/embedding_drift.csv`

---

## 2. Results

### 2.1 Resection vs Soramic

| model_id | input | median_d | mean_d | frac_sig |
|----------|-------|--------:|-------:|---------:|
| `f8aabb75` | bbox | 0.962 | 0.858 | 0.992 |
| `8715461c` | bbox | 0.941 | 0.840 | 1.000 |
| `e12b0592` | bbox | 0.850 | 0.801 | 1.000 |
| `5d04e6ba` | raw  | 0.835 | 0.771 | 0.992 |
| `92b9afed` | bbox | 0.811 | 0.736 | 0.977 |
| `050d401d` | bbox | 0.811 | 0.753 | 1.000 |
| `34e6806f` | raw  | 0.748 | 0.687 | 0.914 |
| `982a6fa2` | raw  | 0.725 | 0.652 | 0.953 |
| `1361bef2` | raw  | 0.725 | 0.687 | 0.984 |
| `a6f970d6` | raw  | 0.683 | 0.676 | 0.977 |
| `9109a6c2` | raw  | 0.677 | 0.666 | 0.977 |
| `a64b245f` | raw  | 0.607 | 0.578 | 0.898 |
| `dc7e1d10` | raw  | 0.598 | 0.601 | 0.945 |
| `6a1a1bdf` | raw  | 0.583 | 0.593 | 0.953 |
| `12e4ba6a` | raw  | 0.545 | 0.562 | 0.984 |
| `5e3f71a0` | raw  | 0.523 | 0.526 | 0.906 |
| `06c598c0` | raw  | 0.503 | 0.519 | 0.922 |

### 2.2 Resection vs Lausanne

| model_id | input | median_d | mean_d | frac_sig |
|----------|-------|--------:|-------:|---------:|
| `f8aabb75` | bbox | 0.947 | 0.844 | 0.984 |
| `5d04e6ba` | raw  | 0.941 | 0.851 | 0.992 |
| `8715461c` | bbox | 0.916 | 0.828 | 1.000 |
| `34e6806f` | raw  | 0.804 | 0.718 | 0.914 |
| `1361bef2` | raw  | 0.784 | 0.715 | 0.961 |
| `e12b0592` | bbox | 0.776 | 0.762 | 1.000 |
| `9109a6c2` | raw  | 0.775 | 0.718 | 0.953 |
| `982a6fa2` | raw  | 0.738 | 0.663 | 0.891 |
| `a6f970d6` | raw  | 0.722 | 0.676 | 0.930 |
| `050d401d` | bbox | 0.714 | 0.702 | 1.000 |
| `6a1a1bdf` | raw  | 0.703 | 0.640 | 0.922 |
| `92b9afed` | bbox | 0.654 | 0.637 | 0.969 |
| `dc7e1d10` | raw  | 0.651 | 0.623 | 0.945 |
| `a64b245f` | raw  | 0.605 | 0.583 | 0.875 |
| `12e4ba6a` | raw  | 0.585 | 0.597 | 0.977 |
| `06c598c0` | raw  | 0.504 | 0.500 | 0.852 |
| `5e3f71a0` | raw  | 0.471 | 0.486 | 0.875 |

### 2.3 Soramic vs Lausanne (drift between the two test cohorts)

| model_id | input | median_d | mean_d | frac_sig |
|----------|-------|--------:|-------:|---------:|
| `dc7e1d10` | raw  | 0.267 | 0.280 | 0.703 |
| `92b9afed` | bbox | 0.260 | 0.268 | 0.727 |
| `5d04e6ba` | raw  | 0.260 | 0.252 | 0.727 |
| `a64b245f` | raw  | 0.245 | 0.263 | 0.617 |
| `6a1a1bdf` | raw  | 0.242 | 0.250 | 0.617 |
| `06c598c0` | raw  | 0.234 | 0.252 | 0.539 |
| `34e6806f` | raw  | 0.231 | 0.235 | 0.594 |
| `5e3f71a0` | raw  | 0.227 | 0.245 | 0.570 |
| `9109a6c2` | raw  | 0.223 | 0.220 | 0.562 |
| `1361bef2` | raw  | 0.215 | 0.229 | 0.531 |
| `982a6fa2` | raw  | 0.208 | 0.213 | 0.500 |
| `a6f970d6` | raw  | 0.194 | 0.211 | 0.430 |
| `12e4ba6a` | raw  | 0.175 | 0.191 | 0.305 |
| `050d401d` | bbox | 0.170 | 0.173 | 0.227 |
| `e12b0592` | bbox | 0.158 | 0.162 | 0.180 |
| `8715461c` | bbox | 0.137 | 0.133 | 0.023 |
| `f8aabb75` | bbox | 0.136 | 0.137 | 0.023 |

### 2.4 Soramic drift vs Lausanne drift (sorted by Δ = lausanne_d − soramic_d)

| model_id | input | lausanne_d | soramic_d | lausanne_sig | soramic_sig | Δ(lau−sor) |
|----------|-------|----------:|----------:|-------------:|------------:|-----------:|
| `6a1a1bdf` | raw | 0.703 | 0.583 | 0.922 | 0.953 | +0.120 |
| `5d04e6ba` | raw | 0.941 | 0.835 | 0.992 | 0.992 | +0.106 |
| `9109a6c2` | raw | 0.775 | 0.677 | 0.953 | 0.977 | +0.098 |
| `1361bef2` | raw | 0.784 | 0.725 | 0.961 | 0.984 | +0.059 |
| `34e6806f` | raw | 0.804 | 0.748 | 0.914 | 0.914 | +0.056 |
| `dc7e1d10` | raw | 0.651 | 0.598 | 0.945 | 0.945 | +0.053 |
| `12e4ba6a` | raw | 0.585 | 0.545 | 0.977 | 0.984 | +0.040 |
| `a6f970d6` | raw | 0.722 | 0.683 | 0.930 | 0.977 | +0.038 |
| `982a6fa2` | raw | 0.738 | 0.725 | 0.891 | 0.953 | +0.013 |
| `06c598c0` | raw | 0.504 | 0.503 | 0.852 | 0.922 | +0.001 |
| `a64b245f` | raw | 0.605 | 0.607 | 0.875 | 0.898 | −0.002 |
| `f8aabb75` | bbox | 0.947 | 0.962 | 0.984 | 0.992 | −0.015 |
| `8715461c` | bbox | 0.916 | 0.941 | 1.000 | 1.000 | −0.025 |
| `5e3f71a0` | raw | 0.471 | 0.523 | 0.875 | 0.906 | −0.052 |
| `e12b0592` | bbox | 0.776 | 0.850 | 1.000 | 1.000 | −0.074 |
| `050d401d` | bbox | 0.714 | 0.811 | 1.000 | 1.000 | −0.097 |
| `92b9afed` | bbox | 0.654 | 0.811 | 0.969 | 0.977 | −0.157 |

---

## 3. Observations

### 3.1 Large drift from resection to both test cohorts

Across all 17 models, median D-statistics are large for both resection→Soramic (0.50–0.96) and resection→Lausanne (0.47–0.95), with 85–100% of embedding dimensions showing significant distributional shift. The training-set embeddings are far from both test cohort distributions for every model configuration.

### 3.2 Soramic and Lausanne are much closer to each other than to resection

Soramic vs Lausanne drift is substantially smaller (median D 0.14–0.27, frac_sig 0.02–0.73) than either cohort's drift from resection. The two test cohort embeddings occupy a similar region of embedding space, distinct from the resection embeddings.

### 3.3 Raw vs bbox input splits the Δ pattern

- **Raw models (Groups 1–3)**: 9 of 10 have Δ ≥ 0, meaning Lausanne embeddings are at least as far from resection as Soramic embeddings. The frozen backbone models (`5e3f71a0`, `a64b245f`) are the exception (Δ ≈ 0 or slightly negative).
- **Bbox models (Group 4)**: All 5 have Δ < 0, meaning their Soramic embeddings are further from resection than their Lausanne embeddings. The largest gap is `92b9afed` (Δ = −0.157).

### 3.4 Lowest-drift frozen models do not consistently outperform

`5e3f71a0` and `06c598c0` (frozen, n=all) have the smallest drift from resection on both test cohorts (median D 0.47–0.52), yet their Soramic AUROCs (0.635, 0.702) and Lausanne AUROCs (0.534, 0.515) span a wide range. Conversely, `1361bef2` has high drift to both cohorts (median D ≈ 0.72–0.78) but achieves Lausanne AUROC = 0.771. Drift magnitude alone does not predict downstream classification performance.
