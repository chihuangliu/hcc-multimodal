"""Generate Venn diagrams for 0504 RNA RFS configs.

Outputs saved to reports/0504/venns/:
  c3_selected_1y.png  — C3 in_cv: per-fold DESeq-selected features (1-year)
  c3_selected_2y.png  — C3 in_cv: per-fold DESeq-selected features (2-year)
  c1_lr_nonzero.png   — C1 before_cv all genes: per-fold LR C=1 non-zero coef features
  c2_lr_nonzero.png   — C2 before_cv predefined: per-fold LR C=1 non-zero coef features
  c1_vs_hcc.png       — C1: preselected features vs 2146 HCC gene set (1y and 2y)
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
C1_DIR = HERE / "deseq_p0.05_before_cv_all_genes"
C2_DIR = HERE / "deseq_p0.05_before_cv_predefined"
C3_DIR = HERE / "deseq_p0.05_in_cv_predefined"
OUT_DIR = HERE / "venns"
OUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
CV_N_FOLDS = 3


# ---------------------------------------------------------------------------
# Load RNA and clinical data (shared across all configs)
# ---------------------------------------------------------------------------
from hcc_multimodal.baselines.data import add_rfs_columns, get_hcc_genes
from hcc_multimodal.baselines.venn_utils import draw_venn2, draw_venn3, save_venn_figure

clinical_data = pd.read_csv(
    ROOT / "data" / "Clinical" / "2025_Nov_18_ICL_Resection_Clinical_Outcome_soramic_format.csv"
).dropna(how="all")
clinical_data = add_rfs_columns(clinical_data)

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

# Full count matrix (no count filter, all genes) — matches C1/C3 no-filter setting
rna_count_all = rna_data.drop(columns=["SID"])
rna_count_all.index = rna_data["SID"]

# HCC-gene-filtered matrix — matches C2/C3 predefined setting
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    hcc_genes = get_hcc_genes()
rna_count_hcc = rna_count_all.loc[:, rna_count_all.columns.intersection(hcc_genes)]
print(f"All genes: {rna_count_all.shape[1]}, HCC genes in matrix: {rna_count_hcc.shape[1]}")

# Merge with clinical
def make_rfs_df(rna_count):
    return rna_count.merge(
        clinical_data[["SID", "rfs_1year", "rfs_2year"]], on="SID"
    ).set_index("SID")

rna_rfs_all = make_rfs_df(rna_count_all)
rna_rfs_hcc = make_rfs_df(rna_count_hcc)


# ---------------------------------------------------------------------------
# Utility: log2(CPM+1) transform — library size from X_full, select feature_names
# ---------------------------------------------------------------------------
def log2_cpm(X_full: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    library_size = np.asarray(X_full.sum(axis=1), dtype=float)
    library_size = np.where(library_size <= 0, 1.0, library_size)
    X_sel = X_full[feature_names]
    cpm = X_sel.div(library_size, axis=0) * 1e6
    return np.log2(cpm + 1.0)


# ---------------------------------------------------------------------------
# Utility: get non-zero LR C=1 features per fold from a pre-transformed X_cv
# ---------------------------------------------------------------------------
def lr_nonzero_per_fold(X_cv: pd.DataFrame, y: pd.Series) -> dict[int, set]:
    """Return {fold: set of non-zero-coef feature names} for LR C=1 L1."""
    skf = StratifiedKFold(n_splits=CV_N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    feature_names = X_cv.columns.tolist()
    result = {}
    for fold_i, (train_idx, _) in enumerate(skf.split(X_cv, y)):
        X_train = X_cv.iloc[train_idx].values
        y_train = y.iloc[train_idx].values
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        X_proc = scaler.fit_transform(imputer.fit_transform(X_train))
        lr = LogisticRegression(
            solver="liblinear", l1_ratio=1.0, C=1, max_iter=1000, random_state=RANDOM_STATE
        )
        lr.fit(X_proc, y_train)
        nonzero_mask = np.abs(lr.coef_[0]) > 0
        result[fold_i + 1] = set(np.array(feature_names)[nonzero_mask])
        n = int(nonzero_mask.sum())
        print(f"  fold {fold_i+1}: {n} non-zero coef features")
    return result


# ===========================================================================
# 1. C3 — per-fold DESeq-selected features
# ===========================================================================
print("\n=== C3: per-fold selected features ===")

for yr, label in [("1y", "1-year"), ("2y", "2-year")]:
    df = pd.read_csv(C3_DIR / f"rfs_{yr}_lr_selected_features.csv")
    fig, ax = plt.subplots(figsize=(6, 5))
    sets = [
        set(df[df.fold == f]["feature"].unique())
        for f in [1, 2, 3]
    ]
    draw_venn3(ax, sets, f"C3 (in_cv, predefined) — {label} RFS\nDESeq-selected features per fold")
    plt.tight_layout()
    save_venn_figure(fig, OUT_DIR / f"c3_selected_{yr}.png")


# ===========================================================================
# 2. C1 — per-fold non-zero LR C=1 coefficient features
# ===========================================================================
print("\n=== C1: non-zero LR C=1 coef features ===")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("C1 (before_cv, all genes) — LR C=1 non-zero coef genes per fold", fontsize=12)

for ax, (rfs_year, yr, label) in zip(axes, [(1, "1y", "1-year"), (2, "2y", "2-year")]):
    # Load preselected features
    presel = pd.read_csv(C1_DIR / f"rfs_{yr}_preselected_features.csv")["feature"].tolist()
    print(f"\nC1 {yr}: {len(presel)} preselected features")
    # Build X_cv: log2(CPM+1) transform using full-gene library size
    X_all_task = rna_rfs_all.drop(columns=["rfs_1year", "rfs_2year"])
    y = rna_rfs_all[f"rfs_{rfs_year}year"]
    mask = ~y.isna()
    X_task, y_task = X_all_task[mask], y[mask]
    X_cv = log2_cpm(X_task, presel)
    fold_sets = lr_nonzero_per_fold(X_cv, y_task)
    sets = [fold_sets[f] for f in [1, 2, 3]]
    draw_venn3(ax, sets, f"{label} RFS")

plt.tight_layout()
save_venn_figure(fig, OUT_DIR / "c1_lr_nonzero.png")


# ===========================================================================
# 3. C2 — per-fold non-zero LR C=1 coefficient features
# ===========================================================================
print("\n=== C2: non-zero LR C=1 coef features ===")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("C2 (before_cv, predefined) — LR C=1 non-zero coef genes per fold", fontsize=12)

for ax, (rfs_year, yr, label) in zip(axes, [(1, "1y", "1-year"), (2, "2y", "2-year")]):
    presel = pd.read_csv(C2_DIR / f"rfs_{yr}_preselected_features.csv")["feature"].tolist()
    print(f"\nC2 {yr}: {len(presel)} preselected features")
    X_hcc_task = rna_rfs_hcc.drop(columns=["rfs_1year", "rfs_2year"])
    y = rna_rfs_hcc[f"rfs_{rfs_year}year"]
    mask = ~y.isna()
    X_task, y_task = X_hcc_task[mask], y[mask]
    X_cv = log2_cpm(X_task, presel)
    fold_sets = lr_nonzero_per_fold(X_cv, y_task)
    sets = [fold_sets[f] for f in [1, 2, 3]]
    draw_venn3(ax, sets, f"{label} RFS")

plt.tight_layout()
save_venn_figure(fig, OUT_DIR / "c2_lr_nonzero.png")


# ===========================================================================
# 4. C1 — preselected features vs HCC gene set
# ===========================================================================
print("\n=== C1: preselected features vs HCC gene set ===")

hcc_set = set(hcc_genes)
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
fig.suptitle("C1 (before_cv, all genes) — DESeq-selected features vs HCC gene set", fontsize=12)

for ax, (yr, label) in zip(axes, [("1y", "1-year"), ("2y", "2-year")]):
    presel = set(pd.read_csv(C1_DIR / f"rfs_{yr}_preselected_features.csv")["feature"].tolist())
    draw_venn2(ax, presel, hcc_set, "C1 selected", "HCC gene set", f"{label} RFS")

plt.tight_layout()
save_venn_figure(fig, OUT_DIR / "c1_vs_hcc.png")

print("\nDone.")
