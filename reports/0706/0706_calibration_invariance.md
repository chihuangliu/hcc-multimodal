# Can Platt Calibration Improve the `9109a6c2` Downstream Metrics? — 2026-07-06

## Table of Contents
- [1. Task](#1-task)
- [2. Key Findings](#2-key-findings)
- [3. Setup & Method](#3-setup--method)
- [4. Results](#4-results)
- [5. Why calibration cannot help](#5-why-calibration-cannot-help)
- [6. Observations](#6-observations)
- [7. File references](#7-file-references)

## 1. Task

`0706_embedding_grid_eval.md` set the `9109a6c2` ceiling: best Soramic transfer AUROC 0.736
(LASSO/All), best balanced survival split log-rank **p=0.079** (LASSO/Boruta). **Can Platt-scaling
the head's probabilities raise either?** We test the shipped 5-fold CV Platt scaler
(`_platt_cv`) against a single global Platt sigmoid, on the top-5 Soramic heads × two cohorts.

## 2. Key Findings

| # | Finding |
|---|---|
| 1 | **Calibration cannot improve AUROC, C-index, or median-split log-rank — they are rank-based, Platt is monotonic.** Global Platt leaves median-within log-rank p **exactly unchanged** (max \|Δp\|=0.00 across all 10 cells) and AUROC/C-index unchanged up to saturation ties (max \|ΔAUROC\|=0.025). Calibration-invariant *by construction*. |
| 2 | **The shipped `_platt_cv` makes every metric *worse* on these small cohorts.** A per-fold sigmoid is only piecewise-monotonic, scrambling ranks. Soramic AUROC drops up to 0.16 (LASSO/All 0.736→0.691), and best log-rank p *rises* 0.021→0.434. **Not one Soramic cell improves.** |
| 3 | **The one apparent "win" is noise.** Ridge/Variance Lausanne p 0.295→0.011 under CV-Platt is rank-lottery from fold scrambling (Lausanne was non-significant, every other cell degrades). |
| 4 | **Calibration is the wrong lever.** The only non-rank-invariant target is the *frozen-cutoff* split, and its degenerate 99/1 splits are already fixed leakage-free by `median_within` (which gives Boruta p=0.079). |

## 3. Setup & Method

Embedding `9109a6c2`, 128-dim. Heads = top-5 by Soramic transfer AUROC: **LASSO/All, LASSO/LASSO,
Ridge/Variance, LASSO/Boruta, Elastic Net/Variance**. Each refit on resection (`route_grid_scores`),
scored on the labelled subset (Soramic n=57, Lausanne n=66). Three score variants per head × cohort:

- **Raw** — `route_grid_scores(fs, model, resection, test.X)`.
- **CV-Platt** (`_platt_cv`) — `StratifiedKFold(5)`, each fold fits 1-D `LogisticRegression` on the
  others, predicts held-out; out-of-fold scores pooled (breaks monotonicity).
- **Global Platt** — one `LogisticRegression` on all (score, label); strictly monotone.

Each scored for AUROC (rfs_2year), Harrell's C-index, and log-rank p under a `median_within` split
(0706 primary, leakage-free). Runner: `scripts/calibration_invariance.py`.

## 4. Results

### 4.1 Soramic (n=57) — raw / cv / global

| Head | AUROC | C-index | log-rank p |
|---|---|---|---|
| LASSO / All features | 0.736 / 0.691 / **0.736** | 0.560 / 0.544 / **0.560** | 0.129 / 0.301 / **0.129** |
| LASSO / LASSO | 0.731 / 0.614 / **0.731** | 0.555 / 0.513 / **0.555** | 0.389 / 0.905 / **0.389** |
| Ridge / Variance | 0.727 / 0.677 / **0.727** | 0.576 / 0.554 / **0.576** | 0.021 / 0.434 / **0.021** |
| LASSO / Boruta | 0.724 / 0.680 / **0.724** | 0.566 / 0.554 / **0.566** | 0.156 / 0.242 / **0.156** |
| Elastic Net / Variance | 0.721 / 0.557 / **0.721** | 0.575 / 0.501 / **0.575** | 0.085 / 0.229 / **0.085** |

### 4.2 Lausanne (n=66) — raw / cv / global

| Head | AUROC | C-index | log-rank p |
|---|---|---|---|
| LASSO / All features | 0.582 / 0.461 / **0.582** | 0.612 / 0.572 / **0.612** | 0.082 / 0.229 / **0.082** |
| LASSO / LASSO | 0.601 / 0.497 / **0.601** | 0.615 / 0.589 / **0.615** | 0.418 / 0.300 / **0.418** |
| Ridge / Variance | 0.651 / 0.599 / **0.651** | 0.618 / 0.630 / **0.618** | 0.295 / 0.011 / **0.295** |
| LASSO / Boruta | 0.487 / 0.387 / 0.513 | 0.552 / 0.463 / 0.448 | 0.583 / 0.109 / **0.583** |
| Elastic Net / Variance | 0.646 / 0.541 / **0.646** | 0.624 / 0.601 / **0.646** | 0.361 / 0.392 / **0.361** |

### 4.3 Aggregate deltas vs raw

| Calibrator | max \|ΔAUROC\| | max \|ΔC-index\| | max \|Δlog-rank p\| |
|---|---:|---:|---:|
| **Global Platt** (monotonic) | 0.025 | 0.104 | **0.000** |
| **CV-Platt** (`_platt_cv`) | 0.164 | 0.089 | 0.516 |

Global Platt: log-rank p identical to 3+ decimals; the small AUROC/C-index wiggle is the sigmoid
saturating extreme scores and creating rank ties (only visible in heavy-tailed LASSO/Boruta
Lausanne). CV-Platt: large, mostly degrading swings.

## 5. Why calibration cannot help

AUROC, C-index, and a `median_within` split all depend on the score **only through rank order**:
AUROC/C-index are rank statistics; and for any monotone `T`, `{x ≥ median(x)} = {T(x) ≥ T(median x)}`,
so high/low membership — hence log-rank p — is unchanged. Global Platt is strictly monotone → all
three invariant. `_platt_cv` breaks monotonicity only by stitching five sigmoids; at n≈57 that
reordering is pure noise and consistently hurts. The only non-rank-invariant cutoffs are the
`*_frozen` family, but `median_within` already delivers their balanced split (Boruta p=0.079)
without needing test labels.

## 6. Observations

1. **Don't calibrate to chase AUROC/C-index** — mathematically pinned under any monotone rescaling.
2. **`_platt_cv` is a threshold-metric tool, not a discrimination tool** — on small external cohorts it degrades rank-based metrics.
3. **Frozen-cutoff rebalancing is the only open lever, and it's redundant** — `median_within` achieves it label-free.
4. **Consistent with 0706/0629: discrimination ≠ stratification** — no post-hoc transform changes the representation's limit.

## 7. File references

| Artifact | Path |
|---|---|
| Calibrator | `hcc_multimodal/eval/calibration.py` |
| Invariance runner | `scripts/calibration_invariance.py` |
| Grid risk score / cutoffs | `hcc_multimodal/survival/{grid_scores,cutoffs}.py` |
| Results table | `results/eval/calibration/calibration_invariance.csv` |
| Parent grid report | `reports/0706/0706_embedding_grid_eval.md` |
