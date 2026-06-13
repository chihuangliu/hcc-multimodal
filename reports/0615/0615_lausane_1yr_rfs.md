# Lausanne — 1-Year RFS Prediction
**Date:** 2026-06-13  
**Covers:** Lausanne cohort  
**Based on:** `reports/0608/0608_ablation_eval_v2.md`

---

## 1. Setup

Two new contrastive models trained on the resection cohort with `outcome_col=rfs_1year`, evaluated on the Lausanne cohort. Downstream head: SelectKBest(f_classif, k=100) + LR or RF on resection embeddings. Multi-lesion: average. MRI: sagittal slices (axis 0).

A 1yr radiomic baseline was also trained on the resection cohort (`models/radiomics/radiomic_rfs_1year_{lr,rf}.joblib`, same pipeline as 2yr).

### Training commands

```bash
# Radiomic baseline
python -m hcc_multimodal.train.train_radiomics --target rfs_1year

# raw, frozen, patient-split
python -m hcc_multimodal.contrastive.train \
  --lam 0.1 --freeze_backbone --n_per_axis all \
  --split-unit patient --mri_type raw \
  --outcome_col rfs_1year --epochs 1 --axes 0
# → e40ffa0b

# bbox, frozen, slice-split
python -m hcc_multimodal.contrastive.train \
  --lam 0.1 --freeze_backbone --n_per_axis all \
  --split-unit slice --mri_type raw_bbox \
  --outcome_col rfs_1year --epochs 50 --axes 0
# → 2eb1f3ca
```

---

## 2. Results — Lausanne

Best head (LR or RF) per model. Multi-lesion: average.

| Config | Model ID | Outcome | LR AUROC | RF AUROC | Best AUROC | 2yr counterpart AUROC |
|--------|----------|---------|---------|---------|-----------|----------------------|
| raw, λ=0.1, frozen, n=all, patient | `e40ffa0b` | 1yr | 0.475 | 0.544 | **0.544** (RF) | 0.534 (LR) |
| bbox, λ=0.1, frozen, n=all, slice | `2eb1f3ca` | 1yr | 0.496 | 0.427 | **0.496** (LR) | 0.614 (RF) |
| Radiomic | — | 1yr | — | 0.569 | 0.557 | **0.569** (LR) | 0.531 (LR) |

---

## 3. File references

| Artifact | Path |
|---|---|
| Radiomic 1yr models | `models/radiomics/radiomic_rfs_1year_{lr,rf}.joblib` |
| Contrastive model (raw, patient) | `training/contrastive/e40ffa0b/` |
| Contrastive model (bbox, slice) | `training/contrastive/2eb1f3ca/` |
| Eval results (raw, patient) | `results/eval/lusanne/all_e40ffa0b_rfs_1year_20260613_165702.json` |
| Eval results (bbox, slice) | `results/eval/lusanne/all_2eb1f3ca_rfs_1year_20260613_192512.json` |
