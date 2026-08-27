# Calibration and the Risk Groups — 2026-08-31

**Head:** A2 — embedding `d7085bf5`, top-3 model ensemble (LASSO/Pearson k=85 + Elastic Net/Pearson
k=43 + L-SVM/Pearson k=43): the survival head behind Table `tab:external_survival`.

**Question.** Supervisor: *"Add calibration for the deployed ensemble — you propose this as risk
stratification, and an uncalibrated risk model can't be used that way."* So: **does recalibrating
the ensemble change the risk groups?**

## Result

Both calibrators are fitted on **resection out-of-fold scores only** (no test labels), then frozen.
The k-means cutoff is re-derived on the recalibrated resection scores and each cohort re-split.

| Cohort | Calibrator | cutoff | high / low | patients moved |
|---|---|--:|--:|--:|
| Resection | *(none — as published)* | 0.463 | 34 / 26 | — |
| Resection | Platt | 0.467 | 34 / 26 | **0** |
| Resection | isotonic | 0.637 | 16 / 44 | 18 |
| SORAMIC | *(none — as published)* | 0.463 | 83 / 17 | — |
| SORAMIC | Platt | 0.467 | 83 / 17 | **0** |
| SORAMIC | isotonic | 0.637 | 77 / 23 | 6 |

The uncalibrated rows reproduce the published splits exactly (34/26 and 83/17), so this is the same
scoring path as `run_restricted.py --freeze-on insample`.

**Under Platt, not one patient changes arm.** Log-rank *p*, HR and RMST are functions of
(time, event, group) alone, and the group vector is identical — so **every row of
`tab:external_survival` is unchanged to every digit**, both cohorts, both endpoints, both horizons.
Nothing needs re-running.

**Under isotonic they do move** — 18 of 60 resection patients, 6 of 100 SORAMIC. And isotonic moves
them for a bad reason: it is not calibrating well in the first place.

### Is the recalibration itself any good?

Calibration slope should be **1**.

| Cohort | Calibrator | slope | AUROC |
|---|---|--:|--:|
| Resection | none | 1.20 | 0.681 |
| Resection | Platt | 0.51 | 0.647 |
| Resection | isotonic | **0.10** | 0.633 |
| SORAMIC | none | 1.19 | 0.722 |
| SORAMIC | Platt | **0.99** | **0.722** |
| SORAMIC | isotonic | **0.11** | 0.677 |

Resection rows refit the calibrator inside a 5-fold CV of the out-of-fold scores, so they are not
read off their own fit. Read SORAMIC as the verdict: it is the external cohort, and at n=54 the
resection rows are noisy (Platt's 0.51 slope there is a small-sample artefact, not a real finding).

**The number that shows isotonic failed is the slope: 0.11 against a target of 1.** Fitted on 54
patients it collapses to five plateaus (0.20, 0.34, 0.50, 0.92, 1.00), squashing the probabilities
into a band too narrow to track risk — a slope of 0.11 is that picture in one number. AUROC agrees:
0.722 → 0.677, lost to the ties its step function creates.

Platt does not fail: slope 0.99, AUROC unchanged. The uncalibrated model was already close (1.19),
so there was little to correct.

Brier, ECE and a Spiegelhalter test point the same way; they are in `calibration_d7085bf5.csv`.

## Why

Platt is **strictly monotone**. The cutoff and the scores pass through the same transform, so
whoever was above the boundary stays above it; re-fitting k-means on the transformed scale only
nudges the boundary (0.463 → 0.467). Risk *stratification* is a ranking + threshold operation, and a
strictly monotone recalibration leaves it untouched by construction.

Isotonic is a step function — weakly monotone. Distinct scores can be mapped to the *same* value,
which breaks the correspondence and lets patients cross the boundary.

**Conclusion:** calibration and the risk groups are separate concerns for this model. Recalibrating
with Platt is free (it changes no survival result) but also unnecessary for the stratification
claim; isotonic is not usable at this sample size.

## Files

| Artifact | Path |
|---|---|
| Runner | `scripts/ensemble_calibration.py` |
| Group-stability table | `results/eval/calibration/ensemble_d7085bf5/stratification_d7085bf5.csv` |
| Per-patient scores | `results/eval/calibration/ensemble_d7085bf5/scores_d7085bf5.csv` |
| Calibration metrics (Brier / ECE / slope, not used above) | `results/eval/calibration/ensemble_d7085bf5/calibration_d7085bf5.csv` |
| Ensemble definition | `results/eval/grid_flat3_bestckpt/d7085bf5/model_ensemble_members.csv` |

```bash
python scripts/ensemble_calibration.py \
  --model-id d7085bf5 \
  --members-csv results/eval/grid_flat3_bestckpt/d7085bf5/model_ensemble_members.csv \
  --cohorts soramic \
  --out-dir results/eval/calibration/ensemble_d7085bf5 \
  --fig-dir reports/0831/calibration
```

Commit `ff2d2dd`.
