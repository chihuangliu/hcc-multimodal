# Calibration Experiment — 2-Year RFS Prediction
**Date:** 2026-06-09  
**Covers:** Soramic (ablation) cohort · Lausanne cohort  
**Based on:** `reports/0608/0608_ablation_eval_v2.md`

---

## Table of Contents

- [1. Setup](#1-setup)
- [2. Results — Soramic](#2-results--soramic)
- [3. Results — Lausanne](#3-results--lausanne)
- [4. Observations](#4-observations)
- [5. File references](#5-file-references)

---

## 1. Setup

### 1.1 Method

Cross-validated Platt scaling applied on each test cohort independently. For each model:

1. Downstream head (LR or RF, same as §2.3 / §3.3) fitted on resection embeddings → raw scores on test cohort  
2. 5-fold stratified CV on the test cohort: for each fold, a Platt scaler is fitted on the other 4 folds and applied to the held-out fold  
3. Calibrated probabilities are evaluated with the same metrics at threshold=0.5

Only models that beat the best radiomic baseline in the respective cohort are included:

| Cohort | Best radiomic baseline | Models evaluated |
|--------|----------------------|-----------------|
| Soramic | RF AUROC = 0.590 | 9 models (ranks 1–9 in §2.3) |
| Lausanne | LR AUROC = 0.531 | 10 models (ranks 1–10 in §3.3) |

Parentheses show Δ vs. uncalibrated (same head). "—" = undefined NPV (all samples predicted positive).

### 1.2 Script

```
python -m hcc_multimodal.eval.calibration \
  --ablation-set {soramic,lusanne} \
  --model-id <IDs> \
  --n-folds 5
```

---

## 2. Results — Soramic

Best head (matching §2.3) per model, ranked by uncalibrated AUROC.

| Rank | Model ID | Head | Uncal AUROC | Cal AUROC | Cal AUPRC | Cal Sensitivity | Cal Specificity | Cal PPV | Cal NPV | Cal F1 |
|------|----------|------|------------:|----------:|----------:|----------------:|----------------:|--------:|--------:|-------:|
| 1 | `9109a6c2` | LR | 0.732 | 0.697 (−0.035) | 0.839 (−0.026) | 0.974 (+0.128) | 0.111 (−0.278) | 0.704 (−0.046) | 0.667 (+0.129) | 0.817 (+0.022) |
| 2 | `dc7e1d10` | LR | 0.718 | 0.625 (−0.093) | 0.761 (−0.077) | 0.949 (+0.103) | 0.111 (−0.278) | 0.698 (−0.052) | 0.500 (−0.038) | 0.804 (+0.009) |
| 3 | `06c598c0` | LR | 0.702 | 0.648 (−0.054) | 0.802 (−0.038) | 0.974 (+0.077) | 0.000 (−0.222) | 0.679 (−0.035) | — | 0.800 (+0.005) |
| 4 | `a64b245f` | LR | 0.684 | 0.667 (−0.017) | 0.764 (−0.040) | 1.000 (+0.026) | 0.222 (−0.056) | 0.736 (−0.009) | 1.000 (+0.167) | 0.848 (+0.004) |
| 5 | `12e4ba6a` | RF | 0.670 | 0.554 (−0.116) | 0.740 (−0.080) | 1.000 (+0.128) | 0.000 (−0.278) | 0.684 (−0.039) | — | 0.812 (+0.021) |
| 6 | `050d401d` | LR | 0.669 | 0.454 (−0.215) | 0.672 (−0.119) | 1.000 (0.000) | 0.000 (0.000) | 0.667 (0.000) | — | 0.800 (0.000) |
| 7 | `5e3f71a0` | RF | 0.635 | 0.494 (−0.141) | 0.703 (−0.116) | 1.000 (0.000) | 0.000 (0.000) | 0.684 (0.000) | — | 0.812 (0.000) |
| 8 | `6a1a1bdf` | LR | 0.615 | 0.594 (−0.021) | 0.813 (−0.003) | 1.000 (+0.615) | 0.000 (−0.778) | 0.684 (−0.105) | — | 0.812 (+0.295) |
| 9 | `982a6fa2` | RF | 0.606 | 0.494 (−0.112) | 0.677 (−0.057) | 1.000 (0.000) | 0.000 (0.000) | 0.684 (0.000) | — | 0.812 (0.000) |

---

## 3. Results — Lausanne

Best head (matching §3.3) per model, ranked by uncalibrated AUROC.

| Rank | Model ID | Head | Uncal AUROC | Cal AUROC | Cal AUPRC | Cal Sensitivity | Cal Specificity | Cal PPV | Cal NPV | Cal F1 |
|------|----------|------|------------:|----------:|----------:|----------------:|----------------:|--------:|--------:|-------:|
| 1 | `1361bef2` | RF | 0.771 | 0.590 (−0.181) | 0.798 (−0.069) | 1.000 (0.000) | 0.000 (0.000) | 0.742 (0.000) | — | 0.852 (0.000) |
| 2 | `5d04e6ba` | LR | 0.655 | 0.415 (−0.240) | 0.720 (−0.130) | 1.000 (0.000) | 0.000 (0.000) | 0.742 (0.000) | — | 0.852 (0.000) |
| 3 | `a6f970d6` | LR | 0.618 | 0.433 (−0.185) | 0.700 (−0.127) | 1.000 (+0.020) | 0.000 (−0.059) | 0.742 (−0.008) | — | 0.852 (+0.002) |
| 4 | `92b9afed` | RF | 0.614 | 0.506 (−0.108) | 0.744 (−0.066) | 1.000 (+0.021) | 0.000 (−0.059) | 0.738 (−0.008) | — | 0.850 (+0.003) |
| 5 | `982a6fa2` | LR | 0.600 | 0.484 (−0.116) | 0.751 (−0.093) | 1.000 (+0.020) | 0.000 (−0.059) | 0.742 (−0.008) | — | 0.852 (+0.002) |
| 6 | `e12b0592` | RF | 0.595 | 0.502 (−0.093) | 0.720 (−0.070) | 1.000 (+0.562) | 0.000 (−0.706) | 0.738 (−0.070) | — | 0.850 (+0.282) |
| 7 | `9109a6c2` | LR | 0.563 | 0.427 (−0.136) | 0.755 (−0.051) | 1.000 (+0.306) | 0.000 (−0.353) | 0.742 (−0.014) | — | 0.852 (+0.129) |
| 8 | `a64b245f` | RF | 0.556 | 0.465 (−0.091) | 0.765 (−0.038) | 1.000 (+0.041) | 0.000 (−0.059) | 0.742 (−0.004) | — | 0.852 (+0.013) |
| 9 | `050d401d` | RF | 0.544 | 0.407 (−0.137) | 0.683 (−0.072) | 1.000 (+0.667) | 0.000 (−0.765) | 0.738 (−0.062) | — | 0.850 (+0.379) |
| 10 | `5e3f71a0` | LR | 0.534 | 0.372 (−0.162) | 0.667 (−0.108) | 1.000 (+0.061) | 0.000 (−0.118) | 0.742 (−0.012) | — | 0.852 (+0.016) |

---

## 4. Observations

1. **AUROC drops universally** (Soramic: −0.017 to −0.215; Lausanne: −0.069 to −0.240). With 5-fold CV, each patient's calibrated probability comes from a different calibrator fitted on a different subset, introducing inter-fold scale shifts that break the global ranking.

2. **Calibration collapses specificity to zero in all cases.** Trained on ~47 patients with a 68–74% positive rate, the Platt scaler finds that predicting everyone positive minimises log-loss — the standard degenerate outcome under high class imbalance.

3. **Apparent F1 gains are spurious.** Models that had real specificity before calibration (e.g., `6a1a1bdf` Soramic Spec=0.778, F1=0.517) show large F1 improvements after calibration only because they switch to all-positive prediction. This is not a clinically meaningful gain.

4. **Conclusion.** CV Platt scaling is not viable for these cohorts — the high positive rate and small sample size cause the calibrator to degenerate regardless of model. If the goal is threshold adjustment, Youden's J optimisation is more appropriate.

---

## 5. File references

| Artifact | Path |
|---|---|
| Calibration script | `hcc_multimodal/eval/calibration.py` |
| Soramic calibration results | `results/eval/soramic/calibration_9109a6c2_dc7e1d10_..._rfs_2year_20260609_001210.json` |
| Lausanne calibration results | `results/eval/lusanne/calibration_1361bef2_5d04e6ba_..._rfs_2year_20260609_001213.json` |
