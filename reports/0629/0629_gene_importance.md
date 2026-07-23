# Gene-Ablation Importance — 2026-06-29

Single-gene leave-one-out importance for the 20-gene `2y_before_cv` set, using
the best Soramic config (run `9109a6c2`). For each gene, the model was retrained
with that one gene dropped (`scripts/submit_gene_ablation.sh`) and evaluated with
`eval --mode embedding` on the Soramic resection→ablation cohort
(`scripts/submit_gene_ablation_eval.sh`). Evaluation committed at `ad88c62`.

- **AUC** = best-head AUROC (max of LR / RF) on the `average` multi-lesion
  embedding, matching the headline metric in the 0622 survival report.
- **Baseline** = full 20-gene run `9109a6c2`, best-head AUROC = **0.732** (LR).
- **Δ AUC** = ablation AUC − baseline. More negative ⇒ the dropped gene
  contributed more (its removal hurt performance most). Sorted most-negative first.
- Cohort: Soramic only (Lausanne ablation evals were not produced). Target `rfs_2year`.

| Gene dropped | AUC | Δ AUC |
|---|---:|---:|
| CSF2 | 0.519 | −0.214 |
| AL449283.1 | 0.530 | −0.202 |
| OR52N5 | 0.537 | −0.195 |
| HIGD2B | 0.546 | −0.187 |
| AL445235.1 | 0.570 | −0.162 |
| AC138647.1 | 0.591 | −0.141 |
| CAMK2N2 | 0.593 | −0.139 |
| RBMXL3 | 0.596 | −0.136 |
| AC093525.8 | 0.603 | −0.130 |
| AC093826.2 | 0.608 | −0.124 |
| H19 | 0.608 | −0.124 |
| AC004241.5 | 0.611 | −0.121 |
| AC025198.1 | 0.630 | −0.103 |
| ZMYND12 | 0.636 | −0.096 |
| AC025580.2 | 0.644 | −0.088 |
| AC130366.1 | 0.651 | −0.081 |
| HNRNPA1P9 | 0.654 | −0.078 |
| SGSM1 | 0.662 | −0.070 |
| AC063947.2 | 0.678 | −0.054 |
| LACC1 | 0.715 | −0.017 |

**Notes**

- Every single-gene removal degrades AUC, so the full 20-gene set is jointly
  informative; no gene is redundant under this metric.
- `CSF2`, `AL449283.1`, and `OR52N5` are the most important (largest drop on
  removal); `LACC1` and `AC063947.2` are the least.
- Best-head AUROC flips between the LR and RF head across runs, which adds noise
  to the ranking. Per-head AUROCs (LR / RF) behind each best-head value:

  | Gene | LR | RF |
  |---|---:|---:|
  | CSF2 | 0.459 | 0.519 |
  | AL449283.1 | 0.383 | 0.530 |
  | OR52N5 | 0.537 | 0.449 |
  | HIGD2B | 0.546 | 0.514 |
  | AL445235.1 | 0.570 | 0.454 |
  | AC138647.1 | 0.591 | 0.515 |
  | CAMK2N2 | 0.476 | 0.593 |
  | RBMXL3 | 0.453 | 0.596 |
  | AC093525.8 | 0.513 | 0.603 |
  | AC093826.2 | 0.608 | 0.492 |
  | H19 | 0.462 | 0.608 |
  | AC004241.5 | 0.611 | 0.347 |
  | AC025198.1 | 0.630 | 0.455 |
  | ZMYND12 | 0.597 | 0.636 |
  | AC025580.2 | 0.644 | 0.353 |
  | AC130366.1 | 0.422 | 0.651 |
  | HNRNPA1P9 | 0.654 | 0.430 |
  | SGSM1 | 0.662 | 0.496 |
  | AC063947.2 | 0.678 | 0.659 |
  | LACC1 | 0.715 | 0.629 |
  | **baseline (9109a6c2)** | **0.732** | **0.568** |
