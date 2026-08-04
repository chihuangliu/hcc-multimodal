# Image Saliency — 2026-08-06

Attributes the deployed 2-year RFS **model ensemble** (`LASSO`/`Pearson` k=85, `Elastic Net`/`Pearson` k=43, `L-SVM`/`Pearson` k=43) on run `d7085bf5` back to MRI voxels, through the frozen image encoder. The main text shows one exemplar per outcome × prediction category (12 patients across `resection`, `soramic`). A further 8 confident hits, taken at the next probability ranks, are in Appendix D.

> **Head.** Both cohorts are scored and attributed by the single downstream head fit on the **whole** resection cohort. Resection `p` is therefore **in-sample and optimistic** — it is not a performance estimate and does not correspond to the cross-validated AUROC in the thesis tables. It is used here only to rank patients into the six categories, so that every panel in this report is attributing the same head. Pass `--resection-head oof` for out-of-fold probabilities with each resection patient attributed by its own held-out fold's head.

## 1. Key findings

1. **The model's strongest evidence usually does not sit on the lesion, and that does not separate hits from misses.** Pooling every patient run, including the Appendix D extras, the most positive slice falls inside the tumour's slice extent for **2 of 12** confident hits (`tp_*`/`tn_*`) and **1 of 8** misses — on Soramic alone, 2 of 6 hits. With only a handful of exemplars per cell this is an observation, not a test; the per-case detail is in Appendix B.
2. **A large share of the extreme slices are at the edge of the volume.** The most negative slice lies in the outermost 10% of the stack for **10 of 20** patients, and the most positive slice for **9 of 20**. Those slices are body wall, air, or a small off-anatomy bright artefact — not liver.
3. **Integrated Gradients needs a blur baseline here.** With a zero baseline the completeness residual does not converge at any practical step count (relative 1.24/4.48/5.46 at 64 steps, still 0.34/0.68/0.73 at 256, non-monotone): a uniform image sits in LayerNorm's near-singular region. The blur baseline converges monotonically to 0.010/0.017/0.034 at 256 steps, which is what §3 reports.
4. **On the resection cohort a large part of the decision magnitude comes from slices that contain no anatomy at all** — a preprocessing artefact carrying 10%–57% of `Σ|c_s|` there and 0.1% or less on Soramic. Mechanism, quantification and cohort asymmetry are in Appendix C.

## 2. Method

The patient embedding is the mean over **every** slice along axis 0 of the volume (`n_per_axis=None`), and all ensemble members are linear, so with `z̄ = (1/S) Σ_s f(x_s)` and `S(z) = (1/M) Σ_m σ(a_m(β_m·z + b_m) + c_m)` the local decision direction `β_eff = ∇_z logit S(z̄)` yields an **exact** additive decomposition

```
c_s = β_eff · f(x_s) / S,        Σ_s c_s = β_eff · z̄
```

Which slices matter is therefore read off the model rather than chosen by hand. The two figures below attribute at the voxel level within those slices: **Integrated Gradients** (256 midpoint steps, baseline `blur`) on the top-16 slices by `|c_s|`, with the most positive and most negative slice forced in, and **Gradient×Input** on every slice, stacked into a 3D volume and projected along the slicing axis. Both are mapped back to the native voxel grid, inverting the backbone's hidden 224→256 resize and 224 centre crop, and rescaled to preserve their signed sum.

The per-slice profile `c_s` itself is plotted in Appendix A; the constant-input slices it exposes, and how they are excluded from the extreme-slice selection, are described in Appendix C.

## 3. Validation gates

- Recomputed `z̄` vs the cached embedding the thesis tables use: max **2.05e-06** over 20 patients
- Decomposition identity `Σ_s c_s` vs `β_eff·z̄`: max **2.24e-05**
- Head reconstruction (closed-form ensemble score vs `predict_proba`): **2.67e-03** — the residual is libsvm's iterative pairwise-coupling step in the Platt-scaled `L-SVM` member, not the unwinding (same behaviour as the gene-side report)
- IG completeness `Σ IG − Δtarget`, each residual against its own slice's target change: worst-patient median **0.047**, worst single slice **0.819** (max absolute 1.17e-03)

## 4. Selected cases

Every patient is attributed with the head fit on all labelled resection patients, so `β_eff` differs between patients only through their own embedding. Resection `p` is in-sample (see the note at the top).

| Cohort | Case | SID | y | p | slices | Σc_s | ‖β_eff‖ | nnz | max-pos slice | max-neg slice | tumour slices |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| resection | tp_high | 162 | 1 | 0.840 | 421 | +0.3404 | 20.5 | 45 | 372 | 407 | 27 |
| resection | tn_low | 49 | 0 | 0.210 | 400 | -1.9087 | 30.5 | 45 | 388 | 333 | 32 |
| resection | fp_high | 186 | 0 | 0.703 | 430 | -0.4528 | 39.7 | 45 | 349 | 145 | 57 |
| resection | fp_borderline | 10 | 0 | 0.506 | 401 | -1.4504 | 49.0 | 45 | 361 | 33 | 86 |
| resection | fn_borderline | 7 | 1 | 0.497 | 473 | -1.4901 | 49.1 | 45 | 308 | 42 | 44 |
| resection | fn_low | 113 | 1 | 0.222 | 440 | -1.9718 | 32.9 | 45 | 192 | 84 | 60 |
| soramic | tp_high | 1905004 | 1 | 0.924 | 450 | +0.8328 | 7.5 | 45 | 352 | 35 | 43 |
| soramic | tn_low | 1201006 | 0 | 0.164 | 420 | -1.7177 | 22.7 | 45 | 418 | 277 | 38 |
| soramic | fp_high | 1001007 | 0 | 0.778 | 345 | -0.0122 | 31.0 | 45 | 15 | 110 | 28 |
| soramic | fp_borderline | 1301008 | 0 | 0.536 | 380 | -1.3136 | 48.3 | 45 | 154 | 27 | 16 |
| soramic | fn_borderline | 1011005 | 1 | 0.480 | 380 | -1.5554 | 49.0 | 45 | 2 | 376 | 90 |
| soramic | fn_low | 1901014 | 1 | 0.231 | 450 | -1.8428 | 31.6 | 45 | 442 | 398 | 44 |

`Σc_s = β_eff·z̄` is the **linear part** of the score at that patient's operating point; the member intercepts carry the rest, so its sign need not track `p`. `max-pos`/`max-neg slice` are the extremes among slices that carry anatomy (Appendix C).

## 5. Figures

### resection

![IG on the extreme slices](image_saliency/top_slices_resection.png)

![Gradient×Input MIP](image_saliency/saliency_mip_resection.png)

### soramic

![IG on the extreme slices](image_saliency/top_slices_soramic.png)

![Gradient×Input MIP](image_saliency/saliency_mip_soramic.png)

## 6. Caveats

- **Spatial resolution.** `vit_b_32` gives a 7×7 patch grid over the 224px input; after the anisotropic resize of an elongated sagittal slice one patch covers tens of millimetres. The attribution is regional, not textural — which is why both figures are pooled to that grid.
- **Frozen backbone** (`freeze_backbone=True`): only the projection MLP was trained, so the spatial features are ImageNet's.
- **Centre crop.** The backbone transform resizes 224→256 and centre-crops back to 224, so a ~6% border of every slice is never seen by the encoder and is given exactly zero attribution.
- **`β_eff` is patient-specific** — the local gradient of a mean of sigmoids. `c_s` decomposes the linearised logit exactly, not `logit S` itself.
- **Per-slice p99 normalisation** means the attribution is with respect to the model's input, not raw MRI intensity.
- **The MIP discards the slice axis.** A positive and a negative peak on the same ray compete, and only the larger survives, so a blank region in the MIP is not evidence of no contribution. Colour scales are per-panel, so saturation is not comparable between patients.
- **Deployed vs training input.** The encoder was trained on the raw resection volumes (resampled on load) but the deployed embeddings are extracted from the preprocessed root without resampling. The attribution follows the **deployed** path — the one behind every number in the thesis tables.

## 7. Regenerate

```
python -m hcc_multimodal.interpretability.image_saliency \
  --model-id d7085bf5 --members-csv results/eval/grid_flat3_bestckpt/d7085bf5/model_ensemble_members.csv \
  --cohorts resection soramic --resection-head full --extra-tp 2 --extra-tn 2 \
  --top-k-slices 16 --ig-steps 256 \
  --output-dir results/eval/interpretability/image_saliency/d7085bf5
python -m hcc_multimodal.interpretability.image_saliency_plots \
  --input-dir results/eval/interpretability/image_saliency/d7085bf5 --fig-dir reports/0810/image_saliency --report reports/0810/0810_image_saliency.md
```

---

## Appendix A — Per-slice contribution profiles

`c_s` against slice index for every case, main and extra. The tumour's extent along the slicing axis is shaded green and constant-input slices (Appendix C) grey. This is the figure that shows *how the score is distributed over the volume* before any voxel-level attribution: the two figures in §5 are zoom-ins on the extremes of these curves.

### resection

![Per-slice contribution profile](image_saliency/slice_profile_resection.png)


### soramic

![Per-slice contribution profile](image_saliency/slice_profile_soramic.png)

## Appendix B — Where the extreme slices sit relative to the tumour

Slice-level overlap only: whether the index of the extreme slice falls within the first and last slice containing tumour. No voxel-level masking is involved.

| Cohort | Case | SID | tumour slice extent | max-pos slice | in tumour? | max-neg slice | in tumour? |
|---|---|---:|---:|---:|:-:|---:|:-:|
| resection | tp_high | 162 | 207–233 | 372 | no | 407 | no |
| resection | tn_low | 49 | 194–323 | 388 | no | 333 | no |
| resection | fp_high | 186 | 295–351 | 349 | **yes** | 145 | no |
| resection | fp_borderline | 10 | 246–331 | 361 | no | 33 | no |
| resection | fn_borderline | 7 | 364–407 | 308 | no | 42 | no |
| resection | fn_low | 113 | 273–332 | 192 | no | 84 | no |
| resection | tp_high_2 | 176 | 254–295 | 5 | no | 329 | no |
| resection | tp_high_3 | 33 | 82–276 | 25 | no | 338 | no |
| resection | tn_low_2 | 133 | 252–326 | 54 | no | 57 | no |
| resection | tn_low_3 | 53 | 204–277 | 202 | no | 341 | no |
| soramic | tp_high | 1905004 | 314–356 | 352 | **yes** | 35 | no |
| soramic | tn_low | 1201006 | 305–342 | 418 | no | 277 | no |
| soramic | fp_high | 1001007 | 213–240 | 15 | no | 110 | no |
| soramic | fp_borderline | 1301008 | 257–272 | 154 | no | 27 | no |
| soramic | fn_borderline | 1011005 | 207–296 | 2 | no | 376 | no |
| soramic | fn_low | 1901014 | 312–355 | 442 | no | 398 | no |
| soramic | tp_high_2 | 1905005 | 189–388 | 229 | **yes** | 40 | no |
| soramic | tp_high_3 | 1013011 | 277–323 | 80 | no | 344 | no |
| soramic | tn_low_2 | 1201008 | 287–311 | 186 | no | 363 | no |
| soramic | tn_low_3 | 1201001 | 249–284 | 2 | no | 14 | no |

Extremes are over slices carrying anatomy (constant-input slices excluded). `c_s` is signed, so a tumour slice can legitimately contribute negatively — the point of the column is *whether the model's strongest evidence sits on the lesion at all*.

## Appendix C — Constant-input (degenerate) slices

`_normalize_slice` (`eval/data.py`, mirrored in `contrastive/data.py`) does `np.clip(s, 0, p99)`. When a slice's 99th percentile is **negative** — a background-only slice of a volume whose background is negative — `a_min > a_max`, so numpy returns `p99` at every pixel, and the following `if p99 > 0` rescale is skipped. The slice reaches the encoder as a **constant image at a large negative value** (~-54 in ImageNet-normalised units, against [-2.12, 2.64] for a real slice): far outside anything the backbone saw in training, carrying no anatomy, and still averaged into the patient embedding.

Such slices are flagged by the mechanism itself (`percentile(slice, 99) <= 0` on the source volume) rather than by testing the tensor for constancy, which is unreliable: two bilinear resizes leave float32 noise on the constant, and the ImageNet normalisation then gives each channel its own constant. They are excluded from the extreme-slice selection — a constant image has no spatial story to tell — but they remain inside the mean-pooled embedding, so their share of `Σ|c_s|` is reported here instead.

| Cohort | Case | SID | slices | degenerate | share of Σ\|c_s\| |
|---|---|---:|---:|---:|---:|
| resection | tp_high | 162 | 421 | 26 (6.2%) | 11.6% |
| resection | tn_low | 49 | 400 | 64 (16.0%) | 38.7% |
| resection | fp_high | 186 | 430 | 168 (39.1%) | 55.8% |
| resection | fp_borderline | 10 | 401 | 41 (10.2%) | 18.9% |
| resection | fn_borderline | 7 | 473 | 95 (20.1%) | 33.5% |
| resection | fn_low | 113 | 440 | 143 (32.5%) | 54.6% |
| resection | tp_high_2 | 176 | 350 | 13 (3.7%) | 10.2% |
| resection | tp_high_3 | 33 | 402 | 38 (9.5%) | 16.5% |
| resection | tn_low_2 | 133 | 400 | 89 (22.2%) | 56.8% |
| resection | tn_low_3 | 53 | 487 | 174 (35.7%) | 50.1% |
| soramic | tp_high | 1905004 | 450 | 1 (0.2%) | 0.1% |
| soramic | tn_low | 1201006 | 420 | 1 (0.2%) | 0.0% |
| soramic | fp_high | 1001007 | 345 | 0 (0.0%) | 0.0% |
| soramic | fp_borderline | 1301008 | 380 | 1 (0.3%) | 0.0% |
| soramic | fn_borderline | 1011005 | 380 | 1 (0.3%) | 0.0% |
| soramic | fn_low | 1901014 | 450 | 1 (0.2%) | 0.0% |
| soramic | tp_high_2 | 1905005 | 450 | 1 (0.2%) | 0.1% |
| soramic | tp_high_3 | 1013011 | 420 | 0 (0.0%) | 0.0% |
| soramic | tn_low_2 | 1201008 | 380 | 1 (0.3%) | 0.0% |
| soramic | tn_low_3 | 1201001 | 380 | 1 (0.3%) | 0.0% |

This is a property of the **deployed** pipeline, not of this analysis: every cached embedding and every AUROC in the thesis was produced with it. It is reported, not fixed — changing `_normalize_slice` would invalidate all of them.

## Appendix D — Additional confident hits

The next-ranked true positives and true negatives by predicted probability. They are kept out of §5 so that each outcome × prediction category is represented there by a single exemplar, but they are what the pooled counts in finding 1 are based on: a pattern seen in one confident hit is not distinguishable from a coincidence.

| Cohort | Case | SID | y | p | slices | Σc_s | ‖β_eff‖ | nnz | max-pos slice | max-neg slice | tumour slices |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| resection | tp_high_2 | 176 | 1 | 0.825 | 350 | +0.1742 | 16.6 | 45 | 5 | 329 | 42 |
| resection | tp_high_3 | 33 | 1 | 0.800 | 402 | +0.1149 | 27.4 | 45 | 25 | 338 | 195 |
| resection | tn_low_2 | 133 | 0 | 0.279 | 400 | -2.0304 | 39.9 | 45 | 54 | 57 | 75 |
| resection | tn_low_3 | 53 | 0 | 0.284 | 487 | -2.0302 | 40.2 | 45 | 202 | 341 | 74 |
| soramic | tp_high_2 | 1905005 | 1 | 0.864 | 450 | +0.3608 | 11.5 | 45 | 229 | 40 | 145 |
| soramic | tp_high_3 | 1013011 | 1 | 0.819 | 420 | +0.2158 | 24.5 | 45 | 80 | 344 | 47 |
| soramic | tn_low_2 | 1201008 | 0 | 0.232 | 380 | -1.8785 | 32.3 | 45 | 186 | 363 | 25 |
| soramic | tn_low_3 | 1201001 | 0 | 0.273 | 380 | -1.9457 | 37.5 | 45 | 2 | 14 | 36 |

### resection

![IG on the extreme slices](image_saliency/top_slices_resection_extra.png)

![Gradient×Input MIP](image_saliency/saliency_mip_resection_extra.png)


### soramic

![IG on the extreme slices](image_saliency/top_slices_soramic_extra.png)

![Gradient×Input MIP](image_saliency/saliency_mip_soramic_extra.png)
