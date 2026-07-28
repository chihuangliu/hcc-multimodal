# Best Config, Randomized Gene Order — 2026-08-03

Six replicates of one config, differing only in the random 40-gene column order.

Each replicate is evaluated at **two checkpoints**: the epoch-10 checkpoint (`<run>@10`,
`epoch_010.pt`) and the run's **best-validation-loss** checkpoint (`best_model.pt`), selected
over a 50-epoch budget with `patience=2`.

| Run | Epochs (10 / best in 50) | Resection CV AUC| Soramic heatmap AUC | Lausanne heatmap AUC| Heatmap Head | τ=24 log-rank p | τ=24 point-p |
|---|---|--:|--:|--:|---|--:|--:|
| `09cd4b36` | 10 | 0.607 | 0.658 | 0.531 | LASSO / RF Import. k=43 | 0.303 | — |
| `18e77da5` | 10 | 0.703 | 0.668 | 0.505 | Elastic Net / Variance k=43 | 0.013 | 0.010 |
| `39d54fe5` | 10 | 0.628 | 0.728 | 0.425 | Ridge / Mutual Info k=85 | 0.153 | 0.025 |
| `d7085bf5` | 10 | 0.632 | 0.598 | 0.454 | Ridge / All features k=all | 0.590 | 0.904 |
| `e924f983` | 10 | 0.628 | 0.689 | 0.424 | L-SVM / LASSO k=85 | 0.196 | 0.096 |
| `6a964bac` | 10 | 0.620 | 0.711 | 0.418 | Elastic Net / Boruta k=43 | 0.021 | 0.016 |
| `09cd4b36` | best (ep 42) | 0.670 | 0.651 | 0.453 | Elastic Net / LASSO k=85 | 0.167 | 0.211 |
| `18e77da5` | best (ep 36) | 0.682 | 0.678 | 0.390 | Ridge / Pearson k=43 | 0.119 | 0.081 |
| `39d54fe5` | best (ep 30) | 0.579 | 0.607 | 0.501 | NB / Boruta k=43 | 0.844 | 0.530 |
| `d7085bf5` | best (ep 42) | 0.694 | 0.694 | 0.429 | LASSO / Pearson k=85 | **0.020** | **0.013** |
| `e924f983` | best (ep 32) | 0.529 | 0.605 | 0.403 | XGB / Mutual Info k=43 | **0.042** | **0.038** |
| `6a964bac` | best (ep 41) | 0.570 | 0.615 | 0.433 | Ridge / Pearson k=43 | 0.525 | 0.514 |

## File references

Epoch-10 artifacts are stored under the bare run id (`.../09cd4b36/`) for five runs and under
`6a964bac__ep010` for the sixth; `best_model.pt` artifacts are under the `_bestckpt` paths.

| | epoch 10 | best_model.pt |
|---|---|---|
| grid | `results/eval/grid_flat3/<run>/` | `results/eval/grid_flat3_bestckpt/<run>/` |
| cv-rank | `results/eval/cv_rank_0803{,_6a964bac}/` | `results/eval/cv_rank_0803_bestckpt/` |
| survival | `results/eval/survival/*_best_<run>_*` | `results/eval/survival/*_bestckpt_<run>_*` |
| heatmaps | `reports/0727/flat3/<run>/` | `reports/0803/flat3_bestckpt/<run>/` |
