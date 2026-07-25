# Reproducing dc7e1d10 with a Deterministic Gene Order — 2026-07-25

`dc7e1d10` (Setting A best encoder in [`0727_embedding_grid_eval_v4.md`](0727_embedding_grid_eval_v4.md))
was trained with a **non-deterministic gene column order** (genes passed as a Python `set`, never
persisted), so its GeneEncoder's gene→slot mapping is unrecoverable and mechanistic gene attribution
is impossible on it. This report retrains `dc7e1d10`'s config under the current code — which pins a
sorted, deterministic gene order — and tests whether the downstream results reproduce, so a valid
interpretable stand-in can be used for gene attribution.

## Method

`dc7e1d10` = base `3e598f36` (5 ep, from scratch) + 5 ep continuation = 10 effective epochs.
Config (both stages): `vit_b_32`, `--freeze_backbone`, `embed_dim 128`, `gene_hidden_dim 256`,
`gene_set all` (40 genes), `n_per_axis all`, `axes 0`, `mri_type raw`, `temperature 0.07`, `lam 0.1`,
`reg_mode per_modality`, `val_split 0.1`, `seed 42`. Two reproduction runs:

| Run | Base | Deterministic order in | Note |
|---|---|---|---|
| `dc7e1d10` | `3e598f36` | — (baseline) | original, non-deterministic |
| **`507d94dc`** | `3e598f36` (original, reused) | last 5 ep only | isolates gene-order change; base held identical |
| **`8f41f04f`** | `98ddd90b` (fresh, 5 ep from scratch) | all 10 ep | fully deterministic chain |

Both reproduction runs are stamped `deterministic_gene_order=True` + the 40-gene sorted order.
Downstream (identical to 0727 v4): image embeddings extracted for resection / Soramic / Lausanne →
flat non-repeated 3-fold resection-CV grid (10 classifiers × 13 FS, `k∈{43,85,128}`) with top-3 model
ensemble → Soramic restricted-time survival (head A1 = Ridge/Variance k=85, `--freeze-on insample`,
best-power cutoff over median/kmeans/youden, τ∈{12,24,36,48}), per the
[`0713_restricted_time_survival_v2.md`](../0713/0713_restricted_time_survival_v2.md) protocol.

## 1. Embedding grid — Setting A

| Metric | `dc7e1d10` (baseline) | `507d94dc` (orig base) | `8f41f04f` (fresh base) |
|---|--:|--:|--:|
| Best-cell resection CV | 0.744 | **0.785** | 0.731 |
| best cell | Ridge/Var k=85 | Ridge/Var k=85 | Ridge/ElasticNet k=43 |
| Anchor LASSO/AllFeat — CV | 0.695 | 0.707 | 0.686 |
| Best-cell — Soramic | 0.709 | 0.654 | 0.567 |
| Top-3 ensemble — Soramic | 0.697 | 0.674 | 0.578 |
| Anchor / best-cell — Lausanne | 0.419 / 0.436 | 0.462 / 0.461 | 0.517 / 0.492 |

## 2. Soramic restricted-time survival — head A1 (Ridge/Variance k=85)

Best-power cutoff sweep (Soramic, full follow-up); all reproduction cutoffs are null:

| Run | picked | hi/lo | τ=24 log-rank | full log-rank | full HR |
|---|---|--:|--:|--:|--:|
| `dc7e1d10` | kmeans | 90 / 10 | **0.043** | 0.114 | 2.25 |
| `507d94dc` | median | 91 / 9 | 0.380 | 0.206 | 0.58 |
| `8f41f04f` | kmeans | 73 / 27 | 0.534 | 0.505 | 0.82 |

Selected-head detail:

| Endpoint / τ | `dc7e1d10` | `507d94dc` | `8f41f04f` |
|---|--:|--:|--:|
| RFS τ=24 — HR / log-rank | 3.93 / **0.043** ✓ | 0.66 / 0.380 | 1.25 / 0.534 |
| RFS τ=24 — point-p | **0.033** ✓ | 0.777 | 0.535 |
| TTR τ=24 — HR / log-rank | 7.26 / **0.024** ✓ | 0.62 / 0.329 | 0.95 / 0.902 |
| C-index (dichotomy) | ~0.52 | ~0.49–0.51 | ~0.48 |

## 3. Success criteria

| # | Criterion | `507d94dc` | `8f41f04f` |
|---|---|:--:|:--:|
| 1 | Resection CV AUC is the highest | ✅ 0.785 (highest of all 3) | ❌ 0.731 (lowest) |
| 2 | Soramic transfer > baseline (~0.71) | ⚠️ anchor 0.712 ≈ baseline; best-cell 0.654 below | ❌ 0.567–0.615, all below |
| 3 | RFS τ=24 significant | ❌ p=0.380 | ❌ p=0.534 |

## Findings

- **Neither deterministic-order run reproduces all three criteria.** `dc7e1d10`'s significant A1
  survival signal (RFS/TTR τ=24 log-rank 0.043 / 0.024) is recovered by neither.
- **Retraining the base from scratch is worse, not better.** The fully-deterministic chain `8f41f04f`
  is the weakest run everywhere (Soramic anchor 0.615, best-cell 0.567; lowest CV). dc7e1d10's
  downstream quality is tied to its **specific frozen base `3e598f36`**, which a same-config, same-seed
  retrain does not reproduce.
- **Gene-order determinism is not what breaks reproduction; the base image encoder is.** `507d94dc`,
  which reuses dc7e1d10's exact original base and only makes the final 5 epochs deterministic,
  reproduces the stable anchor almost exactly (Soramic 0.712 vs 0.718) and has the highest resection CV
  (0.785) — while gaining an interpretable, deterministic gene encoder.
- **The A1 survival split is the most fragile output.** It relies on a sharp 90/10 low arm that exists
  only at dc7e1d10's discrimination level (best-cell Soramic ~0.71); the ~0.05–0.14 drop at that cell in
  both retrains collapses the split to null.

## Conclusion

`507d94dc` is the best deterministic-gene-order stand-in for mechanistic gene attribution: highest
resection CV, Soramic anchor matching baseline, interpretable gene encoder — but it must be documented
as **null on Soramic survival** (criterion 3 not met). A faithful A1 survival reproduction would require
varying the **base image encoder** (seed / epochs), not the gene branch.

## File references

| Artifact | Path |
|---|---|
| Training runs | `training/contrastive/{507d94dc,8f41f04f,98ddd90b}/` (metadata stamped `deterministic_gene_order`) |
| Grid CSVs | `results/eval/grid_flat3/{507d94dc,8f41f04f}/` |
| Survival CSVs | `results/eval/survival/restricted_time_soramic_A1_{507d94dc,8f41f04f}_ridge_var_k85_{rfs,ttr}.csv` |
| Baseline (dc7e1d10) | [`0727_embedding_grid_eval_v4.md`](0727_embedding_grid_eval_v4.md) §4, §6.1 |
| Survival protocol | [`0713_restricted_time_survival_v2.md`](../0713/0713_restricted_time_survival_v2.md) |

Regenerate (both reproduction runs share this shape; `8f41f04f` = fresh base then continuation):
```
# stage: base from scratch (omit --base_model);  continuation: add --base_model <BASE>
python -m hcc_multimodal.contrastive.train --freeze_backbone --n_per_axis all --axes 0 \
  --gene_set all --mri_type raw --epochs 5 --seed 42 [--base_model <BASE>]
# extract → grid → survival (per FINAL run id)
python -m hcc_multimodal.eval.eval --mode embedding --model-id <FINAL> --ablation-set soramic --target rfs_2year
python -m hcc_multimodal.eval.eval --mode embedding --model-id <FINAL> --ablation-set lusanne --target rfs_2year
python -m hcc_multimodal.eval.embedding_grid_eval --task grid --model-id <FINAL> \
  --cv-mode flat --outer-folds 3 --cv-repeats 1 --select-k-fracs 0.333 0.667 1.0 \
  --model-ensemble --model-ensemble-top-k 3 \
  --output-dir results/eval/grid_flat3/<FINAL> --fig-dir reports/0727/flat3/<FINAL>
python -m hcc_multimodal.survival.run_restricted --model-id <FINAL> --fs Variance --model Ridge \
  --select-k 85 --freeze-on insample --select-cutoff-by-power \
  --cutoffs median_frozen kmeans_frozen youden_frozen --taus 12 24 36 48 --no-resection \
  --output-dir results/eval/survival --tag A1_<FINAL>_ridge_var_k85_rfs
```
