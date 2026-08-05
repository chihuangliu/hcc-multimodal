# Embedding Distribution Drift v2 — KS Test on the Deployed Encoder — 2026-08-10

Successor to [`0615_embedding_distribution_drift.md`](../0615/0615_embedding_distribution_drift.md).
Two changes:

- Instead of sweeping 17 contrastive encoders, this report runs the KS analysis on the **single
  deployed encoder** — `d7085bf5`, the Setting A encoder of
  [`0803_embedding_grid_eval_v5.md`](../0803/0803_embedding_grid_eval_v5.md) (raw · slice-level
  split · λ=0.1, read at `best_model.pt`). None of the 17 encoders in the 0615 pool is the deployed
  one.
- Each cohort is restricted to the patients with a **2-year RFS label**, i.e. the exact patients the
  downstream head is fitted and evaluated on. The 0615 report used all cached patients.

## 1. Setup

Patient-level image embeddings (128-dim, mean-pooled over sagittal slices) were read from the
`d7085bf5` cache — the same `resection_img_emb.parquet` /
`ablation_{cohort}_img_emb_raw.parquet` extraction used by the 0803 v5 grid, so these are the exact
vectors the downstream ensemble consumes — then intersected on `SID` with the non-missing
`rfs_2year` labels:

| Cohort | Role | Cached | With 2-yr RFS label | Positives |
|---|---|---:|---:|---:|
| Resection | training | 60 | **54** | 26 (48%) |
| Soramic | ablation test | 100 | **57** | 39 (68%) |
| Lausanne | external test | 68 | **66** | 49 (74%) |

These counts match the AUROC cohorts of 0803 v5 §2 exactly.

Downstream performance of this encoder for reference (0803 v5 §4.2, top-3 model ensemble):
resection CV 0.719, Soramic 0.722, **Lausanne 0.432**.

## 2. Method

For each of the three cohort pairs, a two-sample Kolmogorov–Smirnov test was run **independently on
each of the 128 embedding dimensions**, giving 128 D-statistics and 128 p-values per pair. Reported
per pair:

| Statistic | Definition |
|---|---|
| `median_d` / `mean_d` / `max_d` | median / mean / max KS D-statistic across the 128 dimensions |
| `frac_sig` | fraction of dimensions with raw p < 0.05 |
| `frac_sig_bh` | fraction of dimensions with Benjamini–Hochberg q < 0.05 (128 tests) |

The test is univariate and applied per dimension; embedding dimensions are correlated, so
`frac_sig` counts dimensions, not independent findings.

Each comparison is run twice: once on the pooled cohorts, and once **within each label stratum**
(recurrence within 2 years vs not), which asks whether the drift is shared by both classes or
specific to one.

Script: `hcc_multimodal/eval/embedding_drift.py`
Output: `results/eval/embedding_drift_d7085bf5_rfs2y.csv`

```
python -m hcc_multimodal.eval.embedding_drift \
  --model-id d7085bf5 --input raw --target rfs_2year --stratify-by-label \
  --out results/eval/embedding_drift_d7085bf5_rfs2y.csv
```

## 3. Results

### 3.1 Pooled

| Comparison | n / n | median_d | mean_d | max_d | frac_sig | frac_sig_bh |
|---|---:|---:|---:|---:|---:|---:|
| Resection vs Soramic | 54 / 57 | 0.842 | 0.809 | 0.965 | 1.000 | 1.000 |
| Resection vs Lausanne | 54 / 66 | 0.908 | 0.857 | 1.000 | 1.000 | 1.000 |
| Soramic vs Lausanne | 57 / 66 | 0.299 | 0.287 | 0.492 | 0.641 | 0.586 |

### 3.2 By label stratum

`median_d` only; the significance columns are saturated in every resection comparison
(126–127 of 128 dimensions at BH q < 0.05) and carry no information there.

| Comparison | Positives (n / n) | median_d | Negatives (n / n) | median_d |
|---|---:|---:|---:|---:|
| Resection vs Soramic | 26 / 39 | 0.821 | 28 / 18 | 0.909 |
| Resection vs Lausanne | 26 / 49 | 0.921 | 28 / 17 | 0.964 |
| Soramic vs Lausanne | 39 / 49 | 0.355 | 18 / 17 | 0.275 |

## 4. Observations

1. **Both test cohorts are far from the training cohort, and every dimension is shifted.** All 128
   dimensions reach BH q < 0.05 for both resection comparisons, with median D of 0.84 (Soramic) and
   0.91 (Lausanne) — in the typical dimension the two empirical CDFs are separated over ~85–90% of
   their range. One resection-vs-Lausanne dimension reaches D = 1.000, i.e. the two cohorts'
   values do not overlap at all on that coordinate.

2. **Lausanne is farther than Soramic, but only slightly.** Δ(median D) = +0.066, on a scale where
   both comparisons are already near-total separation. Marginal drift from the training distribution
   does not distinguish the cohort that transfers (Soramic, AUROC 0.722) from the one that does not
   (Lausanne, 0.432).

3. **The two test cohorts resemble each other far more than either resembles resection.**
   Soramic vs Lausanne median D is 0.299 with 59% of dimensions significant after BH — about a third
   of the drift seen from resection. This reproduces the pattern the 0615 report found across all
   17 encoders.

4. **The drift is present in both label strata, and larger for Lausanne in both.** Lausanne exceeds
   Soramic by Δ median D = +0.100 among the positives (0.821 → 0.921) and +0.055 among the negatives
   (0.909 → 0.964), against +0.066 pooled. Within each cohort pair the negative stratum drifts more
   than the positive one, but those two strata hold only 17–18 test patients each (see Caveats).

Label filtering raises all three pooled D-statistics slightly relative to the same run on all cached
patients (0.798 / 0.892 / 0.268 → 0.842 / 0.908 / 0.299) and does not change any of the above.

## 5. Caveats

- KS is univariate and per-dimension; it detects marginal shift and says nothing about whether the
  shifted directions are the ones the downstream head reads. The ensemble uses only the
  Pearson-selected subsets (k = 85 / 43 / 43), not all 128 dimensions.
- The cohorts also differ clinically — on the same 54/57/66 patients, Lausanne's median lesion is
  19 mm against resection's 35 mm, no Lausanne patient is beyond BCLC stage A against 85% of the
  resection cohort, and 28.8% are Child-Pugh B against 3.9%
  ([`0810_cohort_difference.md`](0810_cohort_difference.md)) — so scanner/protocol shift and
  case-mix shift are not separable in these numbers.
- No per-dimension effect size is calibrated against sample size here: at n = 54/57/66 both a D of
  0.30 and a D of 0.91 are easily significant, and `frac_sig` saturates on the resection comparisons.
- The negative strata are small (Soramic 18, Lausanne 17), so their KS null floor is higher than the
  positive strata's (approximately 0.30 vs 0.25 under identical distributions). The positive-minus-
  negative ordering *within* a comparison is therefore partly a sample-size artefact; the
  Lausanne-minus-Soramic differences in §4.4 are not, since those compare like strata.
- D saturates near 1.0. All four resection rows in §3.2 already reach max_d = 1.000 on at least one
  dimension, so differences between 0.82 and 0.96 should not be read as a calibrated ratio of shift
  magnitude. A non-saturating distance (standardised Wasserstein, energy distance) would be needed
  for that.

## 6. File references

| Artifact | Path |
|---|---|
| Script | `hcc_multimodal/eval/embedding_drift.py` |
| Output CSV | `results/eval/embedding_drift_d7085bf5_rfs2y.csv` |
| All-patient counterpart | `results/eval/embedding_drift_d7085bf5.csv` (same command without `--target`) |
| Embedding cache | `training/contrastive/d7085bf5/cached_embeddings/` |
| Encoder + downstream numbers | [`0803_embedding_grid_eval_v5.md`](../0803/0803_embedding_grid_eval_v5.md) §2, §4 |
| Prior 17-encoder sweep | [`0615_embedding_distribution_drift.md`](../0615/0615_embedding_distribution_drift.md) |
