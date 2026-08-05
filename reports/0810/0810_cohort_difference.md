# Cohort Differences on the Evaluated Patients — 2026-08-10

Companion to [`0810_embedding_distribution_drift_v2.md`](0810_embedding_distribution_drift_v2.md).
That report measures the drift in embedding space; this one asks what the three cohorts differ in
clinically, as a side-on read on why transfer holds on Soramic (2-year RFS AUROC 0.722) and fails on
Lausanne (0.432).

The thesis clinical tables describe the **full** cohorts (60 / 104 / 68). The AUROC is computed on
the intersection of "has a cached image embedding" and "has a 2-year RFS label" — **54 / 57 / 66** —
which is a smaller and, for Soramic, a differently composed set. Everything below is tabulated on
that evaluated set.

## 1. Method

Clinical rows were taken from the three CRF exports, restricted to the SIDs entering the
`d7085bf5` evaluation, and summarised as median [IQR] or percentage of recorded values.

```
python -m hcc_multimodal.eval.cohort_characteristics \
  --model-id d7085bf5 --input raw --target rfs_2year
```

Script: `hcc_multimodal/eval/cohort_characteristics.py`
Output: `results/eval/cohort_characteristics.csv`

**Encoding caveat.** The three exports mostly share a scheme (1 = yes / A, 2 = no / B), but
`BCLC Stage` does not: Soramic and Lausanne use `1=0, 2=A, 3=B, 4=C` while the resection export is
offset by one (`1=A, 2=B, 3=C`). Deriving BCLC from the raw column without that offset shifts the
resection cohort by a whole stage. Aetiology is coded 1/blank in the ablation exports and 1/0 in the
resection export, so blanks are counted as absent.

## 2. Results

| Variable | Resection | Soramic | Lausanne |
|---|---|---|---|
| n entering evaluation | 54 | 57 | 66 |
| 2-year RFS positive | 48.1% (26/54) | 68.4% (39/57) | 74.2% (49/66) |
| RFS_central, months | 24.2 [9.6–52.0] | 15.0 [4.2–29.0] | 11.6 [3.5–24.1] |
| Child-Pugh points | 5 [5–6] | 5 [5–6] | 6 [5–7] |
| Child-Pugh points ≥ 7 | 3.9% (2/51) | 17.5% (10/57) | 28.8% (19/66) |
| Child-Pugh class B | 3.9% (2/51) | 14.0% (8/57) | 28.8% (19/66) |
| BCLC beyond stage A | 85.2% (46/54) | 24.6% (14/57) | **0.0% (0/65)** |
| Max lesion diameter, mm | 35.0 [23.8–54.2] | 27.0 [20.0–34.0] | 19.0 [14.2–25.8] |
| Number of lesions | 1 [1–2] | 1 [1–2] | 1 [1–2] |
| Age, years | 65.4 [56.3–69.9] | 66.0 [60.0–72.0] | 65.2 [58.5–71.2] |
| Male | 88.9% (48/54) | 86.0% (49/57) | 86.4% (57/66) |
| Aetiology: alcohol | 22.2% (12/54) | 49.1% (28/57) | 75.8% (50/66) |
| Aetiology: hepatitis B | 22.2% (12/54) | 8.8% (5/57) | 15.2% (10/66) |
| Aetiology: hepatitis C | 22.2% (12/54) | 21.1% (12/57) | 31.8% (21/66) |
| Liver cirrhosis | not recorded | 93.0% (53/57) | 98.5% (65/66) |
| Distinct recruiting sites | not recorded | 12 | not recorded |
| Distinct countries | not recorded | 6 | not recorded |

### 2.1 Label availability

| Cohort | Cached embeddings | 2-yr RFS labelled | Entering evaluation | Median RFS_central, whole file |
|---|---:|---:|---:|---:|
| Resection | 60 | 62 | 54 | 22.5 mo |
| Soramic | 100 | 59 | 57 | 9.3 mo |
| Lausanne | 68 | 66 | 66 | 11.8 mo |

## 3. Observations

**Six variables place Soramic between resection and Lausanne, monotonically**: 2-year recurrence
rate, median RFS, Child-Pugh points ≥ 7, Child-Pugh class B, maximum lesion diameter, and alcohol
aetiology. Hepatitis C is weakly in the same direction (22.1% → 21.1% → 31.8%).

**BCLC stage is the sharpest of them, and is a support-overlap statement rather than a distance
one.** The resection cohort is 85% beyond stage A; Soramic retains 24.6% of such patients; Lausanne
contains **none at all**. On the variable the training cohort is most concentrated in, Lausanne has
zero overlap with it.

**Lesion size has a mechanism specific to this encoder.** The deployed configuration is the raw-crop
encoder (`raw, λ=0.1, slice`), not the bounding-box crop, so the fraction of each image occupied by
tumour scales with lesion diameter. Lausanne's median lesion is 19 mm against resection's 35 mm.

**Not every variable agrees.** Age (65.4 / 66.0 / 65.2), sex (86–89% male) and lesion count
(1 [1–2] in all three) are flat across cohorts, and hepatitis B places Lausanne *closer* to
resection than Soramic (22.2% / 8.8% / 15.2%).

**Soramic is multi-centre; the other two are single-institution cohorts.** Its 57 evaluated patients
carry 12 distinct `Site` codes across 6 countries. The resection and Lausanne exports leave `Site`
and `Country` blank, so this is read from the cohort definitions rather than from the column: both
are single-centre series. For an image encoder this is
a distinct mechanism from disease severity: a multi-centre cohort's embedding distribution is a
mixture over acquisition protocols, so some of its mass can sit near the training site's, while a
single foreign centre contributes one offset with no such component.

**The Soramic evaluation set is selected; the Lausanne one is not.** Soramic caches 100 embeddings
but only 57 carry a 2-year label — its whole-file median RFS_central is 9.3 months, so most patients
lack the follow-up to be labelled. Lausanne keeps 66 of 68. The Soramic test set is therefore
enriched for patients who either recurred early or were followed for two full years, which is a
selection difference independent of any imaging property.

## 4. Caveats

- The six monotone clinical variables are not six independent findings. Liver function, stage,
  lesion size and aetiology are all correlated expressions of one axis — how advanced and how
  decompensated the disease is — and Soramic sitting in the middle is one fact stated six ways.
- With three cohorts and three AUROCs, any monotone variable reproduces the ordering. Nothing here
  can be tested against the outcome, so these are observations consistent with the transfer pattern,
  not an explanation of it. The multi-centre composition (§3) and the label selection (§2.1) are the
  two that propose a mechanism distinct from disease severity.
- `Liver Cirrhosis`, `ECOG`, `Portal Vein Infiltration` and prior-treatment history are unrecorded
  in the resection export, so they cannot be compared to the training cohort at all.
- AUROC is invariant to class prevalence, so the differing positive rates (48% / 68% / 74%) do not
  bias it directly — but Lausanne holds only 17 negatives, which makes its 0.432 a noisy estimate
  regardless of any drift.

## 5. File references

| Artifact | Path |
|---|---|
| Script | `hcc_multimodal/eval/cohort_characteristics.py` |
| Output CSV | `results/eval/cohort_characteristics.csv` |
| Embedding drift companion | [`0810_embedding_distribution_drift_v2.md`](0810_embedding_distribution_drift_v2.md) |
| Encoder + AUROCs | [`0803_embedding_grid_eval_v5.md`](../0803/0803_embedding_grid_eval_v5.md) §2, §4 |
| Full-cohort clinical tables | `hcc-multimodal-thesis/main.tex` Tables 3.1–3.3 |
