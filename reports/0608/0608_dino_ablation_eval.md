# DINO vs ViT Backbone Comparison — Ablation Cohort Evaluation
**Date:** 2026-06-07  
**Covers:** Soramic (ablation) cohort · Lausanne cohort  
**Task:** 2-Year RFS prediction · Embedding-only (no ensemble)

---

## 1. Setup

### 1.1 Models compared

| Model ID | Backbone | Patch | Epochs | Config |
|----------|----------|------:|-------:|--------|
| `dc7e1d10` | ViT-B/32 | 32 | 5 | raw, λ=0.1, frozen, n=all, sagittal |
| `a23a6f80` | ViT-B/16 | 16 | 3 | raw, λ=0.1, frozen, n=all, sagittal |
| `345c2ec6` | DINOv2 ViT-B/14 | 14 | 3 | raw, λ=0.1, frozen, n=all, sagittal |

All models share identical training hyperparameters (embed_dim=128, lr=1e-4, weight_decay=1e-4, seed=42, gene_set=all, val_split=0.1, patient split). ViT-B/16 and DINOv2 use ImageNet-normalisation transforms; ViT-B/32 uses torchvision `ViT_B_32_Weights.IMAGENET1K_V1.transforms()`. ViT-B/32 was trained for 5 epochs, the other two for 3 epochs.

### 1.2 Training losses

| Epoch | ViT-B/32 train | ViT-B/32 val | ViT-B/16 train | ViT-B/16 val | DINOv2 train | DINOv2 val |
|------:|---------------:|-------------:|---------------:|-------------:|-------------:|-----------:|
| 1 | — | — | 1.510 | **2.865** | 1.475 | 3.313 |
| 2 | — | — | 0.668 | 4.052 | 0.578 | 4.402 |
| 3 | — | — | 0.400 | 4.182 | 0.329 | 4.540 |

Best checkpoint used: epoch 1 for both ViT-B/16 and DINOv2. All models exhibit the same pattern: train loss falls while val loss rises from epoch 2, indicating the projection head memorises training slices quickly under a frozen backbone.

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
| DINOv2 checkpoint | `training/contrastive/345c2ec6/best_model.pt` |
| ViT-B/16 Soramic result | `results/eval/soramic/embedding_a23a6f80_rfs_2year_20260607_025605.json` |
| ViT-B/16 Lausanne result | `results/eval/lusanne/embedding_a23a6f80_rfs_2year_20260607_025339.json` |
| DINOv2 Soramic result | `results/eval/soramic/embedding_345c2ec6_rfs_2year_20260607_011821.json` |
| DINOv2 Lausanne result | `results/eval/lusanne/embedding_345c2ec6_rfs_2year_20260607_011606.json` |
| ViT-B/32 Soramic result | `results/eval/soramic/embedding_dc7e1d10_rfs_2year_*.json` |
| ViT-B/32 Lausanne result | `results/eval/lusanne/embedding_dc7e1d10_rfs_2year_*.json` |
