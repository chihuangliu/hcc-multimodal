# DINOv2 / DINOv3 Size Variations — Patient-Split Evaluation
**Date:** 2026-06-09  
**Covers:** Soramic (ablation) cohort · Lausanne cohort  
**Task:** 2-Year RFS prediction · Embedding-only  
**Based on:** `0608_dino_ablation_eval.md` (DINOv2-B baseline reused from `345c2ec6`)

---

## 1. Setup

### 1.1 Models

| Model ID | Backbone | Patch | HF model ID | feat_dim | Params | Epochs | Config |
|----------|----------|------:|-------------|------:|-------:|-------:|--------|
| `99984a4d` | DINOv2 ViT-S/14 | 14 | `facebook/dinov2-small` | 384 | 22M | 1 | raw, λ=0.1, frozen, n=all, sagittal, split=patient |
| `345c2ec6` | DINOv2 ViT-B/14 | 14 | `facebook/dinov2-base` | 768 | 86M | 3† | raw, λ=0.1, frozen, n=all, sagittal, split=patient |
| `2e7acf75` | DINOv2 ViT-L/14 | 14 | `facebook/dinov2-large` | 1024 | 307M | 1 | raw, λ=0.1, frozen, n=all, sagittal, split=patient |
| `0b8ebc50` | DINOv3 ViT-S/16 | 16 | `facebook/dinov3-vits16-pretrain-lvd1689m` | 384 | 22M | 1 | raw, λ=0.1, frozen, n=all, sagittal, split=patient |
| `99c7dedf` | DINOv3 ViT-B/16 | 16 | `facebook/dinov3-vitb16-pretrain-lvd1689m` | 768 | 86M | 1 | raw, λ=0.1, frozen, n=all, sagittal, split=patient |
| `85f467a1` | DINOv3 ViT-L/16 | 16 | `facebook/dinov3-vitl16-pretrain-lvd1689m` | 1024 | 307M | 1 | raw, λ=0.1, frozen, n=all, sagittal, split=patient |

† `345c2ec6` trained for 3 epochs; best checkpoint at epoch 1 (val loss rises thereafter). All new models trained for 1 epoch.

All models: embed_dim=128, lr=1e-4, weight_decay=1e-4, seed=42, gene_set=all, val_split=0.1. ImageNet normalisation for all HF backbones.

### 1.2 Training losses (epoch 1)

| Model ID | Backbone | train | val |
|----------|----------|------:|----:|
| `99984a4d` | DINOv2 ViT-S/14 | 1.778 | 1.936 |
| `345c2ec6` | DINOv2 ViT-B/14 | 1.475 | 3.313 |
| `2e7acf75` | DINOv2 ViT-L/14 | 1.589 | 3.361 |
| `0b8ebc50` | DINOv3 ViT-S/16 | 1.758 | 2.115 |
| `99c7dedf` | DINOv3 ViT-B/16 | 1.665 | 3.155 |
| `85f467a1` | DINOv3 ViT-L/16 | 1.603 | 4.127 |

Val loss at epoch 1 scales with model capacity (larger backbones project into a 128-dim space from a higher-dimensional feature, increasing initial loss). No epoch-2 checks apply (single-epoch runs).

---

## 2. Results — Soramic

| Model ID | Backbone | Head | AUROC | AUPRC | Sensitivity | Specificity | PPV | NPV | F1 |
|----------|----------|------|------:|------:|------------:|------------:|----:|----:|---:|
| `99984a4d` | DINOv2 ViT-S/14 | LR | **0.530** | 0.717 | 0.974 | 0.222 | 0.731 | 0.800 | 0.835 |
| `99984a4d` | DINOv2 ViT-S/14 | RF | 0.395 | 0.645 | 1.000 | 0.000 | 0.684 | — | 0.812 |
| `345c2ec6` | DINOv2 ViT-B/14 | LR | **0.645** | 0.768 | 0.538 | 0.667 | 0.778 | 0.400 | 0.636 |
| `345c2ec6` | DINOv2 ViT-B/14 | RF | 0.546 | 0.723 | 1.000 | 0.000 | 0.684 | — | 0.812 |
| `2e7acf75` | DINOv2 ViT-L/14 | LR | 0.554 | 0.758 | 0.949 | 0.056 | 0.685 | 0.333 | 0.796 |
| `2e7acf75` | DINOv2 ViT-L/14 | RF | **0.620** | 0.781 | 1.000 | 0.000 | 0.684 | — | 0.812 |
| `0b8ebc50` | DINOv3 ViT-S/16 | LR | **0.322** | 0.611 | 0.718 | 0.167 | 0.651 | 0.214 | 0.683 |
| `0b8ebc50` | DINOv3 ViT-S/16 | RF | 0.311 | 0.580 | 1.000 | 0.000 | 0.684 | — | 0.812 |
| `99c7dedf` | DINOv3 ViT-B/16 | LR | **0.537** | 0.683 | 0.282 | 0.722 | 0.688 | 0.317 | 0.400 |
| `99c7dedf` | DINOv3 ViT-B/16 | RF | 0.427 | 0.635 | 0.923 | 0.111 | 0.692 | 0.400 | 0.791 |
| `85f467a1` | DINOv3 ViT-L/16 | LR | **0.597** | 0.748 | 0.821 | 0.222 | 0.696 | 0.364 | 0.753 |
| `85f467a1` | DINOv3 ViT-L/16 | RF | 0.467 | 0.658 | 1.000 | 0.000 | 0.684 | — | 0.812 |

---

## 3. Results — Lausanne

| Model ID | Backbone | Head | AUROC | AUPRC | Sensitivity | Specificity | PPV | NPV | F1 |
|----------|----------|------|------:|------:|------------:|------------:|----:|----:|---:|
| `99984a4d` | DINOv2 ViT-S/14 | LR | 0.364 | 0.709 | 0.918 | 0.059 | 0.738 | 0.200 | 0.818 |
| `99984a4d` | DINOv2 ViT-S/14 | RF | **0.397** | 0.675 | 1.000 | 0.000 | 0.742 | — | 0.852 |
| `345c2ec6` | DINOv2 ViT-B/14 | LR | 0.533 | 0.755 | 0.408 | 0.706 | 0.800 | 0.293 | 0.541 |
| `345c2ec6` | DINOv2 ViT-B/14 | RF | **0.561** | 0.809 | 1.000 | 0.000 | 0.742 | — | 0.852 |
| `2e7acf75` | DINOv2 ViT-L/14 | LR | **0.397** | 0.701 | 0.837 | 0.059 | 0.719 | 0.111 | 0.774 |
| `2e7acf75` | DINOv2 ViT-L/14 | RF | 0.385 | 0.712 | 1.000 | 0.000 | 0.742 | — | 0.852 |
| `0b8ebc50` | DINOv3 ViT-S/16 | LR | 0.502 | 0.771 | 0.673 | 0.353 | 0.750 | 0.273 | 0.710 |
| `0b8ebc50` | DINOv3 ViT-S/16 | RF | **0.519** | 0.769 | 1.000 | 0.000 | 0.742 | — | 0.852 |
| `99c7dedf` | DINOv3 ViT-B/16 | LR | **0.537** | 0.783 | 0.102 | 0.941 | 0.833 | 0.267 | 0.182 |
| `99c7dedf` | DINOv3 ViT-B/16 | RF | 0.478 | 0.727 | 0.735 | 0.294 | 0.750 | 0.278 | 0.742 |
| `85f467a1` | DINOv3 ViT-L/16 | LR | **0.535** | 0.774 | 0.980 | 0.059 | 0.750 | 0.500 | 0.850 |
| `85f467a1` | DINOv3 ViT-L/16 | RF | 0.476 | 0.749 | 1.000 | 0.000 | 0.742 | — | 0.852 |

---

## 4. Summary

Best-head AUROC per model (higher AUROC across LR/RF):

| Model ID | Backbone | Soramic AUROC | Lausanne AUROC | Mean AUROC | Cohort gap |
|----------|----------|-------------:|---------------:|----------:|-----------:|
| `99984a4d` | DINOv2 ViT-S/14 | 0.530 | 0.397 | 0.464 | 0.133 |
| `345c2ec6` | DINOv2 ViT-B/14 | **0.645** | **0.561** | **0.603** | 0.084 |
| `2e7acf75` | DINOv2 ViT-L/14 | 0.620 | 0.397 | 0.509 | 0.223 |
| `0b8ebc50` | DINOv3 ViT-S/16 | 0.322 | 0.519 | 0.421 | −0.197 |
| `99c7dedf` | DINOv3 ViT-B/16 | 0.537 | 0.537 | 0.537 | 0.000 |
| `85f467a1` | DINOv3 ViT-L/16 | 0.597 | 0.535 | 0.566 | 0.062

For reference, ViT-B/32 (`5e3f71a0`, patient-split, λ=0.1): Soramic 0.635, Lausanne 0.534, Mean 0.585, Gap 0.101.

---

## 6. File references

| Artifact | Path |
|---|---|
| DINOv2 ViT-S/14 checkpoint | `training/contrastive/99984a4d/best_model.pt` |
| DINOv2 ViT-B/14 checkpoint (reused) | `training/contrastive/345c2ec6/best_model.pt` |
| DINOv3 ViT-S/16 checkpoint | `training/contrastive/0b8ebc50/best_model.pt` |
| DINOv3 ViT-B/16 checkpoint | `training/contrastive/99c7dedf/best_model.pt` |
| DINOv3 ViT-L/16 checkpoint | `training/contrastive/85f467a1/best_model.pt` |
| DINOv2 ViT-S/14 Soramic result | `results/eval/soramic/embedding_99984a4d_rfs_2year_20260609_032119.json` |
| DINOv2 ViT-S/14 Lausanne result | `results/eval/lusanne/embedding_99984a4d_rfs_2year_20260609_031502.json` |
| DINOv2 ViT-B/14 Soramic result (reused) | `results/eval/soramic/embedding_345c2ec6_rfs_2year_20260607_011821.json` |
| DINOv2 ViT-B/14 Lausanne result (reused) | `results/eval/lusanne/embedding_345c2ec6_rfs_2year_20260607_011606.json` |
| DINOv3 ViT-S/16 Soramic result | `results/eval/soramic/embedding_0b8ebc50_rfs_2year_20260609_032616.json` |
| DINOv3 ViT-S/16 Lausanne result | `results/eval/lusanne/embedding_0b8ebc50_rfs_2year_20260609_031925.json` |
| DINOv3 ViT-B/16 Soramic result | `results/eval/soramic/embedding_99c7dedf_rfs_2year_20260609_034946.json` |
| DINOv3 ViT-B/16 Lausanne result | `results/eval/lusanne/embedding_99c7dedf_rfs_2year_20260609_034234.json` |
| DINOv3 ViT-L/16 Soramic result | `results/eval/soramic/embedding_85f467a1_rfs_2year_20260609_043737.json` |
| DINOv3 ViT-L/16 Lausanne result | `results/eval/lusanne/embedding_85f467a1_rfs_2year_20260609_042930.json` |
| DINOv2 ViT-L/14 checkpoint | `training/contrastive/2e7acf75/best_model.pt` |
| DINOv2 ViT-L/14 Soramic result | `results/eval/soramic/embedding_2e7acf75_rfs_2year_20260609_170925.json` |
| DINOv2 ViT-L/14 Lausanne result | `results/eval/lusanne/embedding_2e7acf75_rfs_2year_20260609_170030.json` |
