# Best Config, Randomized Gene Order — 2026-08-03

Six replicates of one config, differing only in the random 40-gene column order.

| Run | Resection CV AUC| Soramic heatmap AUC | Lausanne heatmap AUC| Heatmap Head | τ=24 log-rank p | τ=24 point-p |
|---|--:|--:|--:|---|--:|--:|
| `09cd4b36` | 0.607 | 0.658 | 0.531 | LASSO / RF Import. k=43 | 0.303 | — |
| `18e77da5` | 0.703 | 0.668 | 0.505 | Elastic Net / Variance k=43 | 0.013 | 0.010 |
| `39d54fe5` | 0.628 | 0.728 | 0.425 | Ridge / Mutual Info k=85 | 0.153 | 0.025 |
| `d7085bf5` | 0.632 | 0.598 | 0.454 | Ridge / All features k=all | 0.590 | 0.904 |
| `e924f983` | 0.628 | 0.689 | 0.424 | L-SVM / LASSO k=85 | 0.196 | 0.096 |
| `6a964bac` | 0.620 | 0.711 | 0.418 | Elastic Net / Boruta k=43 | 0.021 | 0.016 |

**Notes**

- All runs trained for 10 epochs.
