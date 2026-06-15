# DINOv2-Finetuned Backbone in Contrastive Learning

## Task

Test whether initialising the contrastive encoder from a domain-finetuned DINOv2 backbone (`84f180d9`) improves 2-year RFS downstream prediction compared to the baseline contrastive run `345c2ec6` (which starts from the raw ImageNet-pretrained DINOv2).

The new run (`0886b89c`) uses identical contrastive training config to `345c2ec6` (same hyperparameters, 1 epoch, `freeze_backbone=True`) but with `--base_model 84f180d9`.

## Results

| Cohort | Model | AUROC (0886b89c) | AUROC (345c2ec6) | AUPRC (0886b89c) | AUPRC (345c2ec6) |
|--------|-------|-----------------|-----------------|-----------------|-----------------|
| Lausanne | LR | 0.514 | — | 0.792 | — |
| Lausanne | RF | 0.480 | 0.561 | 0.770 | 0.809 |
| Soramic | LR | 0.600 | 0.645 | 0.781 | 0.768 |
| Soramic | RF | 0.632 | — | 0.776 | — |

## Conclusion

The finetuned DINOv2 initialisation does not improve downstream performance. Lausanne RF AUROC drops from 0.561 to 0.480 and Soramic LR AUROC drops from 0.645 to 0.600. AUPRC is mixed (Lausanne worse, Soramic slightly better). The ImageNet-pretrained DINOv2 remains a stronger initialisation for this contrastive setup.
