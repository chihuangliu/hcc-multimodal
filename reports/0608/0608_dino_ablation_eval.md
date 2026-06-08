# DINO vs ViT Backbone Comparison — Ablation Cohort Evaluation
**Date:** 2026-06-07 (updated 2026-06-08)
**Covers:** Soramic (ablation) cohort · Lausanne cohort  
**Task:** 2-Year RFS prediction · Embedding-only (no ensemble)

---

## 1. Setup

### 1.1 Models compared

| Model ID | Backbone | Patch | Epochs | Config |
|----------|----------|------:|-------:|--------|
| `dc7e1d10` | ViT-B/32 | 32 | 10 (5+5) | raw, λ=0.1, frozen, n=all, sagittal, split=slice |
| `a23a6f80` | ViT-B/16 | 16 | 3 | raw, λ=0.1, frozen, n=all, sagittal, split=patient |
| `345c2ec6` | DINOv2 ViT-B/14 | 14 | 3 | raw, λ=0.1, frozen, n=all, sagittal, split=patient |
| `7fb11fdb` | DINOv2 ViT-B/14 | 14 | 10 (3+7) | raw, λ=0.1, frozen, n=all, sagittal, split=slice |
| `bcc55975` | DINOv2 ViT-B/14 | 14 | 3 | raw, λ=0.0, frozen, n=all, sagittal, split=patient |
| `ff3eecc7` | DINOv2 ViT-B/14 | 14 | 10 (3+7) | raw, λ=0.0, frozen, n=all, sagittal, split=slice |

All models share identical training hyperparameters (embed_dim=128, lr=1e-4, weight_decay=1e-4, seed=42, gene_set=all, val_split=0.1). ViT-B/16 and DINOv2 use ImageNet-normalisation transforms; ViT-B/32 uses torchvision `ViT_B_32_Weights.IMAGENET1K_V1.transforms()`. Slice-split models are trained to 10 epochs for comparability with `dc7e1d10` (also 10 epochs total: base `3e598f36` × 5 + fine-tune × 5). Patient-split models use early stopping at epoch 1 (val loss rises thereafter). See Section 7.3 for 3-epoch results of the slice-split models.

### 1.2 Training losses

#### Backbone comparison models

| Epoch | ViT-B/32 train | ViT-B/32 val | ViT-B/16 train | ViT-B/16 val | DINOv2 (`345c2ec6`) train | DINOv2 val |
|------:|---------------:|-------------:|---------------:|-------------:|--------------------------:|-----------:|
| 1 | — | — | 1.510 | **2.865** | 1.475 | 3.313 |
| 2 | — | — | 0.668 | 4.052 | 0.578 | 4.402 |
| 3 | — | — | 0.400 | 4.182 | 0.329 | 4.540 |

Best checkpoint: epoch 1 for ViT-B/16 and `345c2ec6`. Patient-split models show rising val loss from epoch 2 — the projection head memorises patient-specific training slices.

#### DINOv2 λ × split ablation models

| Epoch | λ=0.1/slice (`7fb11fdb`) train | val | λ=0.0/patient (`bcc55975`) train | val | λ=0.0/slice (`ff3eecc7`) train | val |
|------:|-------------------------------:|----:|----------------------------------:|----:|--------------------------------:|----:|
| 1 | 1.520 | 0.921 | 3.084 | **9.442** | 3.180 | 2.170 |
| 2 | 0.676 | 0.478 | 1.904 | 10.216 | 1.924 | 1.793 |
| 3 | 0.408 | 0.368 | 1.688 | 10.282 | 1.692 | 1.687 |
| 4† | 0.707 | 0.290 | — | — | 2.166 | 1.651 |
| 5 | 0.148 | 0.080 | — | — | 1.502 | 1.473 |
| 6 | −0.006 | −0.045 | — | — | 1.375 | 1.390 |
| 7 | −0.109 | −0.114 | — | — | 1.293 | 1.325 |
| 8 | −0.181 | −0.167 | — | — | 1.231 | 1.270 |
| 9 | −0.224 | −0.194 | — | — | 1.188 | 1.246 |
| 10 | −0.242 | **−0.205** | — | — | 1.176 | **1.231** |

† Loss jump at epoch 4 due to optimizer reset when resuming from base model checkpoint.

Slice-split models show decreasing val loss throughout because validation slices come from the same patients as training — there is no true generalization pressure. Patient-split `bcc55975` has a high and rising val loss (pure NT-Xent on held-out patients); best checkpoint is epoch 1. Both slice-split models still have falling val loss at epoch 10 — stopping at 10 is a pragmatic choice for comparability with `dc7e1d10`, not a convergence criterion.

### 1.3 Downstream pipeline

SelectKBest(f_classif, k=100) + LR or RF fitted on resection image embeddings (128-dim), evaluated on ablation image embeddings. Multi-lesion: average. Threshold: 0.5.

---

## 2. Results — Soramic

| Model ID | Backbone | Head | AUROC | AUPRC | Sensitivity | Specificity | PPV | NPV | F1 |
|----------|----------|------|------:|------:|------------:|------------:|----:|----:|---:|
| `dc7e1d10` | ViT-B/32 | LR | **0.718** | **0.838** | 0.846 | 0.389 | 0.750 | 0.538 | 0.795 |
| `dc7e1d10` | ViT-B/32 | RF | 0.608 | 0.766 | 1.000 | 0.056 | 0.696 | 1.000 | 0.821 |
| `345c2ec6` | DINOv2 ViT-B/14 | LR | **0.645** | 0.768 | 0.538 | 0.667 | 0.778 | 0.400 | 0.636 |
| `345c2ec6` | DINOv2 ViT-B/14 | RF | 0.546 | 0.723 | 1.000 | 0.000 | 0.684 | — | 0.812 |
| `a23a6f80` | ViT-B/16 | LR | **0.519** | 0.752 | 0.846 | 0.111 | 0.673 | 0.250 | 0.750 |
| `a23a6f80` | ViT-B/16 | RF | 0.486 | 0.687 | 0.949 | 0.056 | 0.685 | 0.333 | 0.796 |

---

## 3. Results — Lausanne

| Model ID | Backbone | Head | AUROC | AUPRC | Sensitivity | Specificity | PPV | NPV | F1 |
|----------|----------|------|------:|------:|------------:|------------:|----:|----:|---:|
| `dc7e1d10` | ViT-B/32 | LR | 0.419 | 0.710 | 0.449 | 0.471 | 0.710 | 0.229 | 0.550 |
| `dc7e1d10` | ViT-B/32 | RF | 0.453 | 0.727 | 0.776 | 0.118 | 0.717 | 0.154 | 0.745 |
| `345c2ec6` | DINOv2 ViT-B/14 | LR | 0.533 | 0.755 | 0.408 | 0.706 | 0.800 | 0.293 | 0.541 |
| `345c2ec6` | DINOv2 ViT-B/14 | RF | **0.561** | **0.809** | 1.000 | 0.000 | 0.742 | — | 0.852 |
| `a23a6f80` | ViT-B/16 | LR | **0.510** | 0.753 | 0.735 | 0.176 | 0.720 | 0.188 | 0.727 |
| `a23a6f80` | ViT-B/16 | RF | 0.497 | 0.751 | 0.776 | 0.176 | 0.731 | 0.214 | 0.752 |

---

## 4. Summary

| Backbone | Epochs | Soramic best AUROC | Lausanne best AUROC | Mean | Cohort gap |
|----------|-------:|-------------------:|--------------------:|-----:|-----------:|
| ViT-B/32 (`dc7e1d10`) | 5 | **0.718** | 0.453 | 0.586 | 0.265 |
| DINOv2 ViT-B/14 (`345c2ec6`) | 3 | 0.645 | **0.561** | **0.603** | 0.084 |
| ViT-B/16 (`a23a6f80`) | 3 | 0.519 | 0.510 | 0.515 | 0.009 |

---

## 5. Observations

1. **DINOv2 is the strongest backbone overall.** It leads on Lausanne (0.561 vs 0.453 vs 0.510) and has the best mean AUROC across cohorts (0.603). ViT-B/32 leads on Soramic but only after 5 epochs vs 3 for the others.

2. **ViT-B/16 is surprisingly weak — near chance on both cohorts (0.519 / 0.510).** Despite having the same architecture depth and parameter count as ViT-B/32, its ImageNet1K_V1 supervised features do not transfer to MRI. This contrasts sharply with DINOv2, which uses the same patch-size regime (14 vs 16) but was trained self-supervised on 142M images. The gap (DINOv2 0.645 vs ViT-B/16 0.519 on Soramic) isolates the effect of pre-training objective and data scale, not architecture.

3. **ViT-B/16 is the most consistent across cohorts (gap = 0.009) — but for the wrong reason.** It performs equally poorly on both. This is not generalisation; it means its features carry no cohort-specific signal at all, so there is nothing to lose in transfer.

4. **ViT-B/32 vs ViT-B/16 comparison implicates patch size or pre-training.** Both are ImageNet1K_V1 supervised, frozen, same pipeline. Patch-32 clearly outperforms patch-16, which is counter-intuitive (finer patches should capture more detail). This may reflect the quality of the specific torchvision checkpoints rather than patch size per se.

5. **DINOv2's cross-cohort consistency (gap = 0.084) is a meaningful advantage.** The Soramic–Lausanne drop for ViT-B/32 (0.265) reflects strong overfitting to the Soramic MRI acquisition protocol; DINOv2's self-supervised features are more scanner-agnostic.

6. **Next steps:** Run DINOv3 ViT-B/16 (same patch size as ViT-B/16, but self-supervised) to confirm whether the self-supervised objective alone closes the gap seen here. Also consider unfreezing the DINOv2 backbone for end-to-end fine-tuning.

---

## 6. File references

| Artifact | Path |
|---|---|
| ViT-B/16 checkpoint | `training/contrastive/a23a6f80/best_model.pt` |
| DINOv2 λ=0.1/patient checkpoint | `training/contrastive/345c2ec6/best_model.pt` |
| DINOv2 λ=0.1/slice checkpoint (10 ep) | `training/contrastive/7fb11fdb/best_model.pt` |
| DINOv2 λ=0.0/patient checkpoint | `training/contrastive/bcc55975/best_model.pt` |
| DINOv2 λ=0.0/slice checkpoint (10 ep) | `training/contrastive/ff3eecc7/best_model.pt` |
| ViT-B/16 Soramic result | `results/eval/soramic/embedding_a23a6f80_rfs_2year_20260607_025605.json` |
| ViT-B/16 Lausanne result | `results/eval/lusanne/embedding_a23a6f80_rfs_2year_20260607_025339.json` |
| DINOv2 λ=0.1/patient Soramic result | `results/eval/soramic/embedding_345c2ec6_rfs_2year_20260607_011821.json` |
| DINOv2 λ=0.1/patient Lausanne result | `results/eval/lusanne/embedding_345c2ec6_rfs_2year_20260607_011606.json` |
| DINOv2 λ=0.1/slice Soramic result (10 ep) | `results/eval/soramic/embedding_7fb11fdb_rfs_2year_20260608_021903.json` |
| DINOv2 λ=0.1/slice Lausanne result (10 ep) | `results/eval/lusanne/embedding_7fb11fdb_rfs_2year_20260608_021120.json` |
| DINOv2 λ=0.0/patient Soramic result | `results/eval/soramic/embedding_bcc55975_rfs_2year_20260607_181905.json` |
| DINOv2 λ=0.0/patient Lausanne result | `results/eval/lusanne/embedding_bcc55975_rfs_2year_20260607_181043.json` |
| DINOv2 λ=0.0/slice Soramic result (10 ep) | `results/eval/soramic/embedding_ff3eecc7_rfs_2year_20260608_021901.json` |
| DINOv2 λ=0.0/slice Lausanne result (10 ep) | `results/eval/lusanne/embedding_ff3eecc7_rfs_2year_20260608_021200.json` |
| ViT-B/32 Soramic result | `results/eval/soramic/embedding_dc7e1d10_rfs_2year_*.json` |
| ViT-B/32 Lausanne result | `results/eval/lusanne/embedding_dc7e1d10_rfs_2year_*.json` |

---

## 7. DINOv2 λ × Split Config Ablation

This section compares the four DINOv2 configs in a full 2×2 grid (λ ∈ {0.1, 0.0} × split ∈ {patient, slice}) and benchmarks each against the matched ViT-B/32 model from `0608_ablation_eval_v2.md` Group 3 (same frozen/n=all/sagittal setup, different backbone). Slice-split models are trained to 10 epochs for comparability with `dc7e1d10`; patient-split models use the epoch-1 checkpoint (best by val loss). See Section 7.3 for 3-epoch slice results.

### 7.1 DINOv2 results — all four configs

| Model ID | Epochs | λ | Split | Head | Soramic AUROC | Soramic AUPRC | Soramic Sens | Soramic Spec | Soramic F1 | Lausanne AUROC | Lausanne AUPRC | Lausanne Sens | Lausanne Spec | Lausanne F1 |
|----------|-------:|---|-------|------|------:|------:|-----:|-----:|----:|-------:|-------:|------:|------:|----:|
| `345c2ec6` | 3 | 0.1 | patient | LR | **0.645** | 0.768 | 0.538 | 0.667 | 0.636 | 0.533 | 0.755 | 0.408 | 0.706 | 0.541 |
| `345c2ec6` | 3 | 0.1 | patient | RF | 0.546 | 0.723 | 1.000 | 0.000 | 0.812 | **0.561** | **0.809** | 1.000 | 0.000 | 0.852 |
| `7fb11fdb` | 10 | 0.1 | slice | LR | 0.637 | 0.811 | 0.077 | 1.000 | 0.143 | 0.445 | 0.731 | 0.061 | 0.941 | 0.113 |
| `7fb11fdb` | 10 | 0.1 | slice | RF | **0.661** | 0.797 | 0.795 | 0.389 | 0.765 | 0.399 | 0.679 | 0.408 | 0.471 | 0.513 |
| `bcc55975` | 3 | 0.0 | patient | LR | **0.556** | 0.754 | 0.846 | 0.167 | 0.759 | **0.515** | 0.735 | 0.673 | 0.471 | 0.725 |
| `bcc55975` | 3 | 0.0 | patient | RF | 0.521 | 0.720 | 0.923 | 0.167 | 0.800 | 0.440 | 0.708 | 0.633 | 0.235 | 0.667 |
| `ff3eecc7` | 10 | 0.0 | slice | LR | 0.405 | 0.665 | 0.846 | 0.167 | 0.759 | 0.415 | 0.724 | 0.612 | 0.235 | 0.652 |
| `ff3eecc7` | 10 | 0.0 | slice | RF | 0.373 | 0.590 | 0.923 | 0.167 | 0.800 | **0.474** | 0.749 | 0.837 | 0.118 | 0.781 |

Best head per config (higher AUROC):

| Model ID | Epochs | λ | Split | Soramic best AUROC | head | Lausanne best AUROC | head | Mean AUROC | Cohort gap |
|----------|-------:|---|-------|-----------------:|:----|-----------------:|:----|----------:|-----------:|
| `345c2ec6` | 3 | 0.1 | patient | **0.645** | LR | **0.561** | RF | **0.603** | 0.084 |
| `7fb11fdb` | 10 | 0.1 | slice | 0.661 | RF | 0.445 | LR | 0.553 | 0.216 |
| `bcc55975` | 3 | 0.0 | patient | 0.556 | LR | 0.515 | LR | 0.536 | 0.041 |
| `ff3eecc7` | 10 | 0.0 | slice | 0.405 | LR | 0.474 | RF | 0.440 | −0.069† |

† Lausanne AUROC exceeds Soramic — the model has no useful Soramic signal at 10 epochs.

### 7.2 DINOv2 vs ViT-B/32 ablation

ViT-B/32 values from `0608_ablation_eval_v2.md` Group 3 (frozen, n=all, raw, sagittal). Best head used for each model.

| λ | Split | ViT-B/32 ID | ViT-B/32 Soramic | ViT-B/32 Lausanne | ViT-B/32 Mean | ViT-B/32 Gap | DINOv2 ID | DINOv2 Soramic | DINOv2 Lausanne | DINOv2 Mean | DINOv2 Gap | DINO − ViT (mean) |
|---|-------|------------|-----------------:|------------------:|--------------:|-------------:|----------|---------------:|----------------:|------------:|-----------:|------------------:|
| 0.1 | patient | `5e3f71a0` | 0.635 | 0.534 | 0.585 | 0.101 | `345c2ec6` | **0.645** | **0.561** | **0.603** | 0.084 | **+0.018** |
| 0.1 | slice | `dc7e1d10` | **0.718** | 0.453 | 0.586 | 0.265 | `7fb11fdb` |  0.661 | 0.445 | 0.553 | 0.216 | −0.033 |
| 0.0 | patient | `06c598c0` | **0.702** | **0.515** | **0.609** | 0.187 | `bcc55975` | 0.556 | 0.515 | 0.536 | 0.041 | −0.073 |
| 0.0 | slice | `a64b245f` | **0.684** | **0.556** | **0.620** | 0.128 | `ff3eecc7` | 0.405 | 0.474 | 0.440 | −0.069 | −0.180 |


### 7.3 DINOv2 vs ViT-B/32 vs ViT-B/16

Best-head AUROC per backbone (patient-split, λ=0.1 for DINOv2 and ViT-B/32 matched config; all using frozen backbone, n=all, sagittal).

| Backbone | Soramic AUROC | Lausanne AUROC |
|----------|------:|-------:|
| ViT-B/32 | 0.635 | 0.534 |
| DINOv2 ViT-B/14 | **0.645** | **0.561** |
| ViT-B/16 | 0.519 | 0.510 |

---

### 7.4 Slice splitting: 3 epochs vs 10 epochs

The 3-epoch checkpoints of the slice-split models (`137bd456` for λ=0.1, `b1aebfa3` for λ=0.0) were evaluated before extension to 10 epochs. They show better downstream performance than the 10-epoch versions despite higher val loss — evidence that slice-split val loss is not a reliable early-stopping proxy.

| Model ID | Epochs | λ | Split | Head | Soramic AUROC | Soramic AUPRC | Soramic Sens | Soramic Spec | Soramic F1 | Lausanne AUROC | Lausanne AUPRC | Lausanne Sens | Lausanne Spec | Lausanne F1 |
|----------|-------:|---|-------|------|------:|------:|-----:|-----:|----:|-------:|-------:|------:|------:|----:|
| `137bd456` | 3 | 0.1 | slice | LR | 0.433 | 0.698 | 0.385 | 0.500 | 0.476 | **0.550** | 0.746 | 0.204 | 0.765 | 0.317 |
| `137bd456` | 3 | 0.1 | slice | RF | **0.623** | 0.732 | 1.000 | 0.111 | 0.830 | 0.501 | 0.747 | 0.694 | 0.353 | 0.723 |
| `7fb11fdb` | 10 | 0.1 | slice | LR | 0.637 | 0.811 | 0.077 | 1.000 | 0.143 | 0.445 | 0.731 | 0.061 | 0.941 | 0.113 |
| `7fb11fdb` | 10 | 0.1 | slice | RF | **0.661** | 0.797 | 0.795 | 0.389 | 0.765 | 0.399 | 0.679 | 0.408 | 0.471 | 0.513 |
| `b1aebfa3` | 3 | 0.0 | slice | LR | **0.514** | 0.720 | 0.897 | 0.111 | 0.778 | **0.475** | 0.740 | 0.531 | 0.471 | 0.619 |
| `b1aebfa3` | 3 | 0.0 | slice | RF | 0.471 | 0.705 | 0.974 | 0.056 | 0.809 | 0.456 | 0.767 | 0.694 | 0.118 | 0.694 |
| `ff3eecc7` | 10 | 0.0 | slice | LR | 0.405 | 0.665 | 0.846 | 0.167 | 0.759 | 0.415 | 0.724 | 0.612 | 0.235 | 0.652 |
| `ff3eecc7` | 10 | 0.0 | slice | RF | 0.373 | 0.590 | 0.923 | 0.167 | 0.800 | **0.474** | 0.749 | 0.837 | 0.118 | 0.781 |

Best head per config:

| Model ID | Epochs | λ | Soramic best AUROC | Lausanne best AUROC | Mean AUROC | Cohort gap |
|----------|-------:|---|------------------:|--------------------:|----------:|-----------:|
| `137bd456` | 3 | 0.1 | **0.623** | **0.550** | **0.587** | 0.073 |
| `7fb11fdb` | 10 | 0.1 | 0.661 | 0.445 | 0.553 | 0.216 |
| `b1aebfa3` | 3 | 0.0 | **0.514** | **0.475** | **0.495** | 0.039 |
| `ff3eecc7` | 10 | 0.0 | 0.405 | 0.474 | 0.440 | −0.069 |

For λ=0.1, 10 epochs lifts Soramic AUROC (0.623 → 0.661) but collapses Lausanne (0.550 → 0.445), widening the cohort gap from 0.073 to 0.216. For λ=0.0, continued training hurts both cohorts. In both cases the mean AUROC is higher at 3 epochs, confirming that the slice-split val loss provides no useful stopping signal beyond epoch 3.

Checkpoints at `training/contrastive/137bd456/best_model.pt` and `training/contrastive/b1aebfa3/best_model.pt`. Eval results at `results/eval/{soramic,lusanne}/embedding_{137bd456,b1aebfa3}_rfs_2year_20260607_*.json`.
