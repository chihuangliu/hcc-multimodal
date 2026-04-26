# Clinical Summary — ICL Resection Cohort
**Dataset:** `2025_Nov_18_ICL_Resection_Clinical_Outcome_soramic_format.csv`  
**N** = 69 patients, 189 columns; 35 columns have <50% missing data.

---

## Demographics

| Feature | Value |
|---|---|
| Age (years) | mean 62.8, median 64.7, range 27.7–88.2 |
| Sex | "1" 60 (87%), "2" 9 (13%) |

## ECOG Performance Status
All 0

## HCC Etiology (non-exclusive, n=69)

| Etiology | N | % |
|---|---|---|
| Hepatitis B | 16 | 23.2% |
| Hepatitis C | 15 | 21.7% |
| Alcohol | 18 | 26.1% |
| NASH | 10 | 14.5% |

---

## Tumour Characteristics

### Number of Lesions
| Lesions | N | % |
|---|---|---|
| 1 | 49 | 71.0% |
| 2 | 6 | 8.7% |
| 3 | 7 | 10.1% |
| 4 | 1 | 1.4% |
| 5 | 6 | 8.7% |

### Max Diameter of Largest Lesion (mm)
| Stat | Value |
|---|---|
| Min | 3 mm |
| Median | 35 mm |
| Mean | 43.6 mm |
| Max | 130 mm |

### BCLC Stage
| Stage | N | % |
|---|---|---|
| 1 | 10 | 14.5% |
| 2 | 47 | 68.1% |
| 3 | 12 | 17.4% |

### Child-Pugh Class
| Class | N | % |
|---|---|---|
| 1 | 63 | 90.0% |
| 2 | 2 | 2.9% |
| Missing | 5 | 7.1% |

### Child-Pugh points
| Point | N | % |
|---|---|---|
| 5 | 47 | 67.1% |
| 6 | 16 | 22.9% |
| 7 | 2 | 2.9% |
| Missing | 4 | 5.7% |


### Vascular Invasion
| Category | N | % |
|---|---|---|
| 0 | 23 | 32.9% |
| Micro | 39 | 55.7% |
| Micro/Macro | 2 | 2.9% |
| Missing | 6 | 8.6% |

Binary label (`vascular_invasion_label`): 41 positive (1), 23 negative (0).

### Resection Score
| Category | N | % |
|---|---|---|
| R0 (0) | 41 | 58.6% |
| RFA | 13 | 18.6% |
| Macro/Micro | 6 | 8.6% |
| Macro | 2 | 2.9% |
| 0/RFA | 1 | 1.4% |
| Micro | 1 | 1.4% |
| Missing | 6 | 8.6% |

### Tumour Grading
| Grade | N | % |
|---|---|---|
| Well | 4 | 5.7% |
| Moderate | 45 | 64.3% |
| Poor | 7 | 10.0% |
| Necrosis | 1 | 1.4% |
| N/A – no tumour | 1 | 1.4% |
| Other (0) | 1 | 1.4% |
| Missing | 11 | 15.7% |

Binary grade columns:
- **Well:** 17 / 68 (25.0%)
- **Moderate:** 45 / 69 (65.2%)
- **Poor:** 7 / 69 (10.1%)

---

## Survival Endpoints (central assessment)

All times are in **months**.

### Time to Recurrence (TTR_central)
- Evaluable: **69**
- **Event occurred (=1):** n=38, median time-to-recurrence = **11.61 months**
- **Censored (=0):** n=31, median follow-up = **38.30 months**

### Recurrence-Free Survival (RFS_central)
- Evaluable: **69**
- **Event occurred (=1):** n=46, median time-to-event = **14.86 months**
- **Censored (=0):** n=23, median follow-up = **52.73 months**

### Overall Survival (OS_central)
- Evaluable: **64** (6 missing event flag)
- **Event occurred (=1):** n=27, median time-to-death = **49.64 months**
- **Censored (=0):** n=37, median follow-up = **48.00 months**

---

## Excluded Columns

The remaining 154 columns (>50% missing) cover:
- Physical examination findings (ENT, lungs, heart, skin, etc.)
- Vital signs and ECG
- Study arm / treatment assignment (SIRT, sorafenib/placebo)
- TACE/RFA session dates and post-procedure imaging
- Duplicate or investigator-reported survival endpoints (`_original`, `_calc`, `_investigator` variants)
- Metastasis sub-site flags
- Administrative fields (consent, country, site)
