"""Generate per-fold LR coefficient counts and Venn diagrams for DESeq2 + L1 LR pipeline.

Outputs:
  reports/0427/deseq_lr_coef_counts_all.csv     — count summary (all targets, folds, C values)
  reports/0427/deseq_lr_nonzero_genes_all.csv   — gene-level coefficients per fold/model/target
  reports/0427/venn_rna/venn_deseq_lr_c1_1y.png
  reports/0427/venn_rna/venn_deseq_lr_c1_2y.png

Columns in deseq_lr_coef_counts_all.csv:
  target, model, fold, n_selected_deseq, n_nonzero_coef, n_positive_coef, n_negative_coef

Columns in deseq_lr_nonzero_genes_all.csv:
  target, fold, model, gene, coefficient
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib_venn import venn3
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
VENN_DIR = OUT_DIR / "venn_rna"
VENN_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
RNA_COUNT_FILTER = 15
RNA_COUNT_FILTER_MIN_SAMPLES = 5
DESEQ_PVALUE = 0.1
DESEQ_MIN_FEATURES = 20
CV_N_FOLDS = 3
LR_CS = [0.001, 0.01, 0.1, 1]

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
clinical_data = pd.read_csv(
    ROOT / "data" / "Clinical" / "2025_Nov_18_ICL_Resection_Clinical_Outcome_soramic_format.csv"
).dropna(how="all")
rna_data = pd.read_csv(
    ROOT / "data" / "RNA_seq" / "Matrix_output_radiology_only.csv"
).dropna(how="all")

columns = rna_data["Gene Symbol"]
rna_data = rna_data.T.iloc[2:, :]
rna_data.columns = columns.values
rna_data = rna_data.loc[:, ~pd.isnull(rna_data.columns)]
rna_data = rna_data.loc[:, ~rna_data.columns.duplicated()]
rna_data = rna_data.apply(pd.to_numeric, errors="coerce")
rna_data.index = rna_data.index.astype(int)
rna_data = rna_data.rename_axis("SID").reset_index()

from hcc_multimodal.baselines.data import add_rfs_columns
from hcc_multimodal.baselines.evaluation import DeseqCPMSelector

clinical_data = add_rfs_columns(clinical_data)

rna_count = rna_data.drop(columns=["SID"])
rna_count.index = rna_data["SID"]
rna_count_flt = rna_count.loc[
    :, (rna_count >= RNA_COUNT_FILTER).sum(axis=0) >= RNA_COUNT_FILTER_MIN_SAMPLES
]
rna_rfs = rna_count_flt.merge(
    clinical_data[["SID", "rfs_1year", "rfs_2year"]], on="SID"
).set_index("SID")
X_all = rna_rfs.drop(columns=["rfs_1year", "rfs_2year"])

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
count_rows = []
gene_rows = []

# fold_gene_sets[year][c][fold_i] = set of nonzero-coef gene names
fold_gene_sets = {1: {c: {} for c in LR_CS}, 2: {c: {} for c in LR_CS}}

for year in [1, 2]:
    y = rna_rfs[f"rfs_{year}year"]
    mask = ~y.isna()
    X, y_clean = X_all[mask], y[mask]

    outer_cv = StratifiedKFold(n_splits=CV_N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for fold_i, (train_idx, _) in enumerate(outer_cv.split(X, y_clean)):
        X_train = X.iloc[train_idx]
        y_train = y_clean.iloc[train_idx]

        sel = DeseqCPMSelector(pvalue=DESEQ_PVALUE, min_features=DESEQ_MIN_FEATURES)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            X_train_sel = sel.fit_transform(X_train, y_train)

        selected_genes = X.columns[sel.support_].tolist()
        n_selected = len(selected_genes)
        print(f"  year={year} fold={fold_i+1}: {n_selected} genes selected by DESeq2")

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_sel)

        for c in LR_CS:
            lr = LogisticRegression(
                solver="liblinear", penalty="l1", C=c, max_iter=1000, random_state=RANDOM_STATE
            )
            lr.fit(X_train_scaled, y_train)
            coef = lr.coef_[0]
            nonzero_mask = np.abs(coef) > 0
            n_nonzero = int(nonzero_mask.sum())
            nonzero_genes = np.array(selected_genes)[nonzero_mask]
            nonzero_coefs = coef[nonzero_mask]

            fold_gene_sets[year][c][fold_i + 1] = set(nonzero_genes)

            count_rows.append({
                "target": f"rfs_{year}year",
                "model": f"LR_C={c}",
                "fold": fold_i + 1,
                "n_selected_deseq": n_selected,
                "n_nonzero_coef": n_nonzero,
                "n_positive_coef": int((nonzero_coefs >= 0).sum()),
                "n_negative_coef": int((nonzero_coefs < 0).sum()),
            })
            for gene, coef_val in zip(nonzero_genes, nonzero_coefs):
                gene_rows.append({
                    "target": f"rfs_{year}year",
                    "fold": fold_i + 1,
                    "model": f"LR_C={c}",
                    "gene": gene,
                    "coefficient": round(float(coef_val), 6),
                })

# ---------------------------------------------------------------------------
# Save CSVs
# ---------------------------------------------------------------------------
df_counts = pd.DataFrame(count_rows)
df_counts.to_csv(OUT_DIR / "deseq_lr_coef_counts_all.csv", index=False)
print("\nSaved deseq_lr_coef_counts_all.csv")

df_genes = pd.DataFrame(gene_rows)
df_genes.to_csv(OUT_DIR / "deseq_lr_nonzero_genes_all.csv", index=False)
print("Saved deseq_lr_nonzero_genes_all.csv")

print("\nCount summary:")
print(df_counts.to_string(index=False))

# ---------------------------------------------------------------------------
# Venn diagrams — C=1 only (only C with consistent nonzero features)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, year in zip(axes, [1, 2]):
    sets = [fold_gene_sets[year][1].get(f, set()) for f in [1, 2, 3]]
    sizes = [len(s) for s in sets]
    v = venn3(sets, set_labels=(f"Fold 1\n(n={sizes[0]})", f"Fold 2\n(n={sizes[1]})", f"Fold 3\n(n={sizes[2]})"), ax=ax)
    ax.set_title(f"RFS {year}-year LR C=1 — DESeq2-selected, non-zero coef genes", fontsize=11)

    f12 = sets[0] & sets[1]
    f13 = sets[0] & sets[2]
    f23 = sets[1] & sets[2]
    f123 = sets[0] & sets[1] & sets[2]
    print(f"\nRFS {year}y (fold nonzero sizes: {sizes})")
    print(f"  F1∩F2: {sorted(f12) or 'empty'}")
    print(f"  F1∩F3: {sorted(f13) or 'empty'}")
    print(f"  F2∩F3: {sorted(f23) or 'empty'}")
    print(f"  F1∩F2∩F3: {sorted(f123) or 'empty'}")

plt.suptitle("DESeq2 + L1 LR (C=1) — non-zero coefficient genes per fold", fontsize=13)
plt.tight_layout()
venn_path = VENN_DIR / "venn_deseq_lr_c1.png"
plt.savefig(venn_path, bbox_inches="tight", dpi=150)
plt.close()
print(f"\nSaved {venn_path}")
