# DINOv2 vs ViT-B/32 — Ablation Cohort Evaluation
**Date:** 2026-06-07  
**Covers:** Soramic (ablation) cohort · Lausanne cohort  
**Task:** 2-Year RFS prediction · Embedding-only (no ensemble)

---

## 1. Setup

### 1.1 Models compared

| Model ID | Backbone | Epochs | Config |
|----------|----------|-------:|--------|
| `dc7e1d10` | ViT-B/32 | 5 | raw, λ=0.1, frozen, n=all, sagittal slices |
| `345c2ec6` | DINOv2 ViT-B/14 | 3 | raw, λ=0.1, frozen, n=all, sagittal slices |

Both models share identical training hyperparameters (embed_dim=128, lr=1e-4, weight_decay=1e-4, seed=42, gene_set=all, val_split=0.1, patient split). The only differences are the backbone and number of epochs.

### 1.2 DINOv2 training summary (345c2ec6)

| Epoch | Train loss | Val loss |
|------:|-----------:|---------:|
| 1 | 1.4749 | 3.3132 |
| 2 | 0.5777 | 4.4023 |
| 3 | 0.3289 | 4.5403 |

Best checkpoint: epoch 1 (val loss 3.3132). Train loss dropping fast while val loss rises — the projection head overfits quickly with a frozen backbone, matching the ViT-B/32 pattern.

### 1.3 Downstream pipeline

SelectKBest(f_classif, k=100) + LR or RF fitted on resection image embeddings (128-dim), evaluated on ablation image embeddings. Multi-lesion: average. Threshold: 0.5.

---

## 2. Results — Soramic

| Model | Backbone | Head | AUROC | AUPRC | Sensitivity | Specificity | PPV | NPV | F1 |
|-------|----------|------|------:|------:|------------:|------------:|----:|----:|---:|
| `dc7e1d10` | ViT-B/32 | LR | **0.718** | **0.838** | 0.846 | 0.389 | 0.750 | 0.538 | 0.795 |
| `dc7e1d10` | ViT-B/32 | RF | 0.608 | 0.766 | 1.000 | 0.056 | 0.696 | 1.000 | 0.821 |
| `345c2ec6` | DINOv2 ViT-B/14 | LR | **0.645** | 0.768 | 0.538 | 0.667 | 0.778 | 0.400 | 0.636 |
| `345c2ec6` | DINOv2 ViT-B/14 | RF | 0.546 | 0.723 | 1.000 | 0.000 | 0.684 | — | 0.812 |

**Best per model:** ViT-B/32 LR → 0.718 · DINOv2 LR → 0.645 · Δ = −0.073

---

## 3. Results — Lausanne

| Model | Backbone | Head | AUROC | AUPRC | Sensitivity | Specificity | PPV | NPV | F1 |
|-------|----------|------|------:|------:|------------:|------------:|----:|----:|---:|
| `dc7e1d10` | ViT-B/32 | LR | 0.419 | 0.710 | 0.449 | 0.471 | 0.710 | 0.229 | 0.550 |
| `dc7e1d10` | ViT-B/32 | RF | 0.453 | 0.727 | 0.776 | 0.118 | 0.717 | 0.154 | 0.745 |
| `345c2ec6` | DINOv2 ViT-B/14 | LR | 0.533 | 0.755 | 0.408 | 0.706 | 0.800 | 0.293 | 0.541 |
| `345c2ec6` | DINOv2 ViT-B/14 | RF | **0.561** | **0.809** | 1.000 | 0.000 | 0.742 | — | 0.852 |

**Best per model:** ViT-B/32 RF → 0.453 · DINOv2 RF → 0.561 · Δ = +0.108

---

## 4. Summary

| Cohort | ViT-B/32 best AUROC | DINOv2 best AUROC | Δ (DINOv2 − ViT) |
|--------|--------------------:|------------------:|------------------:|
| Soramic | 0.718 | 0.645 | −0.073 |
| Lausanne | 0.453 | 0.561 | **+0.108** |
| Mean | 0.586 | 0.603 | +0.017 |
| Soramic–Lausanne gap | 0.265 | 0.084 | — |

---

## 5. Observations

1. **ViT-B/32 wins on Soramic, DINOv2 wins on Lausanne.** The Soramic advantage for ViT-B/32 (Δ=−0.073) is smaller than the Lausanne advantage for DINOv2 (Δ=+0.108), so DINOv2 comes out slightly ahead on mean AUROC (0.603 vs 0.586).

2. **DINOv2 generalises more consistently across cohorts.** The Soramic–Lausanne gap collapses from 0.265 (ViT-B/32) to 0.084 (DINOv2). ViT-B/32's frozen features transfer well within the Soramic acquisition protocol but fail on the Lausanne `MRI_liver_arterial` scans; DINOv2's patch-14 features trained on a much larger corpus appear more scanner-agnostic.

3. **DINOv2 had only 3 epochs vs 5 for ViT-B/32.** The val loss already climbs from epoch 1, so more epochs would not help without regularisation or unfreezing the backbone. The comparison is fair in terms of what each model converged to, but a longer-trained DINOv2 would not be expected to improve under the current frozen + projection-head-only setup.

4. **LR head favours ViT-B/32 on Soramic; RF head favours DINOv2 on Lausanne.** The LR boundary suits ViT-B/32's more discriminative Soramic features. DINOv2's RF best on Lausanne achieves sensitivity=1.0/specificity=0.0 (all-positive prediction), so the AUROC gain is real but the calibration is poor — the ranking signal is better, but the threshold behaviour is degenerate.

5. **Next steps:** Run DINOv2 with unfrozen backbone or more epochs; evaluate DINOv3 (same config); compare resection CV AUC to check whether DINOv2's better cross-cohort transfer holds in-distribution.

---

## 6. File references

| Artifact | Path |
|---|---|
| DINOv2 checkpoint | `training/contrastive/345c2ec6/best_model.pt` |
| DINOv2 Soramic result | `results/eval/soramic/embedding_345c2ec6_rfs_2year_20260607_011821.json` |
| DINOv2 Lausanne result | `results/eval/lusanne/embedding_345c2ec6_rfs_2year_20260607_011606.json` |
| ViT-B/32 Soramic result | `results/eval/soramic/embedding_dc7e1d10_rfs_2year_*.json` |
| ViT-B/32 Lausanne result | `results/eval/lusanne/embedding_dc7e1d10_rfs_2year_*.json` |
| Training losses | `training/contrastive/345c2ec6/losses.csv` |
