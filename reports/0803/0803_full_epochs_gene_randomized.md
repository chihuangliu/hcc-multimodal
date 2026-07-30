# Best Config, Randomized Gene Order — 2026-08-03

Six replicates of one config, differing only in the random 40-gene column order.

Each replicate is evaluated at **two checkpoints**: the epoch-10 checkpoint (`<run>@10`,
`epoch_010.pt`) and the run's **best-validation-loss** checkpoint (`best_model.pt`), selected
over a 50-epoch budget with `patience=2`.

| Run | Epochs (10 / best in 50) | Resection CV AUC| Soramic heatmap AUC | Lausanne heatmap AUC| Heatmap Head | τ=24 log-rank p | τ=24 point-p |
|---|---|--:|--:|--:|---|--:|--:|
| `09cd4b36` | 10 | 0.607 | 0.658 | 0.531 | LASSO / RF Import. k=43 | 0.303 | — |
|  | best (ep 42) | 0.670 | 0.651 | 0.453 | Elastic Net / LASSO k=85 | 0.167 | 0.211 |
| `18e77da5` | 10 | 0.703 | 0.668 | 0.505 | Elastic Net / Variance k=43 | 0.013 | 0.010 |
|  | best (ep 36) | 0.682 | 0.678 | 0.390 | Ridge / Pearson k=43 | 0.119 | 0.081 |
| `39d54fe5` | 10 | 0.628 | 0.728 | 0.425 | Ridge / Mutual Info k=85 | 0.153 | 0.025 |
|  | best (ep 30) | 0.579 | 0.607 | 0.501 | NB / Boruta k=43 | 0.844 | 0.530 |
| `d7085bf5` | 10 | 0.632 | 0.598 | 0.454 | Ridge / All features k=all | 0.590 | 0.904 |
|  | best (ep 42) | 0.694 | 0.694 | 0.429 | LASSO / Pearson k=85 | **0.020** | **0.013** |
| `e924f983` | 10 | 0.628 | 0.689 | 0.424 | L-SVM / LASSO k=85 | 0.196 | 0.096 |
|  | best (ep 32) | 0.529 | 0.605 | 0.403 | XGB / Mutual Info k=43 | **0.042** | **0.038** |
| `6a964bac` | 10 | 0.620 | 0.711 | 0.418 | Elastic Net / Boruta k=43 | 0.021 | 0.016 |
|  | best (ep 41) | 0.570 | 0.615 | 0.433 | Ridge / Pearson k=43 | 0.525 | 0.514 |

## λ × mri_type × split-unit grid — 2026-07-30

Seven of the eight cells of the `lam` × `mri_type` × `split-unit` grid; the `raw` / `slice` / λ=0.1
cell was not requested and is not run here. All share the hyperparameters above
(vit_b_32, frozen, embed_dim=128, gene_hidden_dim=256, temperature=0.07, reg_mode=per_modality,
gene_set=all, n_per_axis=all over all three axes, 50-epoch budget with `patience=2`,
`checkpoint_interval=10`, bs=32, lr=1e-4, wd=1e-4, seed=42) and differ only in the three grid axes.
Evaluated at **`best_model.pt`** only. Gene order was randomised per run (no `--sort_genes`); each
run's resolved order is in its `metadata.json`.

| Run | Epochs (10 / best in 50) | Resection CV AUC| Soramic heatmap AUC | Lausanne heatmap AUC| Heatmap Head | τ=24 log-rank p | τ=24 point-p |
|---|---|--:|--:|--:|---|--:|--:|
| `78456720` (bbox · slice · λ=0.1) | best (ep 30 of 32) | 0.636 | 0.326 | 0.430 | LASSO / Mutual Info k=43 | 0.848‡ | 0.000‡ |
| `41c6db8a` (bbox · slice · λ=0.0) | best (ep 21 of 23) | 0.579 | 0.500† | 0.500† | NB / RF Import. k=43 | — | — |
| `d33f74db` (bbox · patient · λ=0.1) | best (ep 1 of 3)\* | 0.529 | 0.500† | 0.500† | NB / RFE k=43 | — | — |
| `26a5a902` (bbox · patient · λ=0.0) | best (ep 1 of 3)\* | 0.610 | 0.561 | 0.425 | KNN / Mutual Info k=43 | 0.208 | 0.196 |
| `5cd1cc2d` (raw · patient · λ=0.1) | best (ep 1 of 3)\* | 0.628 | 0.664 | 0.474 | RF / RF Import. k=43 | — | — |
| `a2f950af` (raw · patient · λ=0.0) | best (ep 1 of 3)\* | 0.578 | 0.611 | 0.537 | LR / RFE k=43 | 0.615 | 0.978 |
| `e837a0b4` (raw · slice · λ=0.0) | best (ep 36 of 38) | 0.521 | 0.500† | 0.500† | NB / Elastic Net k=43 | — | — |

\* **The four patient-split runs are single-epoch encoders.** Their validation loss rose
monotonically from epoch 1 (e.g. `d33f74db` 2.462 → 3.924 → 4.697), so `patience=2` stopped them at
epoch 3 and `best_model.pt` is the epoch-1 weights, while train loss kept falling — overfitting
against a held-out-*patient* validation set rather than failing to learn. They had no 50-epoch
budget in any meaningful sense and are **not comparable** to the three slice-split rows, which
trained 23–38 epochs. This is the same split-unit confound seen previously; with `patience=2` it is
fatal rather than merely limiting.

† **Degenerate transfer — not a score.** Naive Bayes wins the resection CV on these three runs
(grid CV 0.603 / 0.694 / 0.719 — including the two highest grid-CV cells of the seven) but its
posteriors saturate: on both external cohorts it predicts every patient positive (sensitivity 1.0,
specificity 0.0), so the AUROC is
exactly 0.500 by tie-breaking and carries no ranking information. 13–24 of the 130 cells are
degenerate this way per run. Because every score lands on one side of the resection-frozen
threshold, all three cutoffs (median/kmeans/youden) leave an **empty low arm** and τ=24 is
undefined — hence the em-dashes. `5cd1cc2d` is a different failure: `RF / RF Import.` ranks
acceptably (Soramic 0.664) but all 100 Soramic scores still sit above every frozen cutoff, so its
split is 100/0 and τ=24 is likewise undefined.

‡ `78456720`'s cutoff carves a **1-patient** low arm (97/1) under all three strategies; the log-rank
is uninformative and the 0.000 point-p is a degenerate single-patient variance estimate, not signal.

**Read.** Only two rows are genuinely evaluable end-to-end — `26a5a902` (79/19, p=0.208) and
`a2f950af` (53/47, p=0.615) — and both are **null**. No cell of this grid reaches the τ=24
significance of `d7085bf5` (0.020) or `e924f983` (0.042) in the table above, and the resection CV
AUCs here (0.521–0.636) sit entirely below the 0.694 top of the best-checkpoint rows above,
overlapping only their weaker half (0.529–0.694). The best-transferring configuration here is
`5cd1cc2d` (raw · patient · λ=0.1, Soramic 0.664), but it is
an epoch-1 encoder with an undefined survival split. Taken together the grid does **not** identify a
configuration that improves on the existing best config.

Regenerate (training then evaluation; `scripts/run_lam_mri_split_grid_local.sh` runs the seven
configs sequentially on a local GPU/MPS box — these runs were trained that way, not on the HPC):
```
bash scripts/run_lam_mri_split_grid_local.sh
python scripts/eval_contrastive_runs.py \
  --run-ids 78456720 41c6db8a d33f74db 26a5a902 5cd1cc2d a2f950af e837a0b4 \
  --tag-prefix lamgrid --results-subdir grid_flat3_lamgrid \
  --fig-dir reports/0803/flat3_lamgrid
```

## File references

Epoch-10 artifacts are stored under the bare run id (`.../09cd4b36/`) for five runs and under
`6a964bac__ep010` for the sixth; `best_model.pt` artifacts are under the `_bestckpt` paths.

| | epoch 10 | best_model.pt |
|---|---|---|
| grid | `results/eval/grid_flat3/<run>/` | `results/eval/grid_flat3_bestckpt/<run>/` |
| cv-rank | `results/eval/cv_rank_0803{,_6a964bac}/` | `results/eval/cv_rank_0803_bestckpt/` |
| survival | `results/eval/survival/*_best_<run>_*` | `results/eval/survival/*_bestckpt_<run>_*` |
| heatmaps | `reports/0727/flat3/<run>/` | `reports/0803/flat3_bestckpt/<run>/` |

λ × mri_type × split-unit grid artifacts (`best_model.pt` only):

| | path |
|---|---|
| training runs | `training/contrastive/<run>/` (losses.csv, metadata.json incl. `gene_order`) |
| grid | `results/eval/grid_flat3_lamgrid/<run>/` |
| cv-rank | `results/eval/cv_rank_lamgrid/cv_rank_image_only.csv` |
| survival | `results/eval/survival/{cutoff_sweep,restricted_time_soramic,restricted_time_lusanne}_lamgrid_<run>_*` |
| heatmaps | `reports/0803/flat3_lamgrid/<run>/heatmap_{cv_auc,soramic_auroc,lusanne_auroc}.{png,svg}` |
| summary | `results/eval/lamgrid_summary.csv` |
| runners | `scripts/run_lam_mri_split_grid_local.sh`, `scripts/eval_contrastive_runs.py` |
