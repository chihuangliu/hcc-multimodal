"""Feature-selection × classifier benchmark grid on a contrastive embedding.

Reproduces the Radiomics "BvM — AUC" heatmap, but on the ``9109a6c2`` embedding
(the highest-Soramic-AUROC ablation model). For each (feature-selection, classifier)
cell we run nested stratified CV on the resection cohort to pick hyperparameters,
then refit on the full resection cohort and evaluate transfer to Soramic and Lausanne.

Outputs (under ``--output-dir``, default ``results/eval/grid/``):
  * ``grid_cv_auc.csv``            — long: model, fs, cv_auc_mean, cv_auc_std
  * ``grid_cv_auc_matrix.csv``     — models × fs matrix of mean nested-CV AUC
  * ``grid_transfer_<cohort>.csv`` — model, fs, metrics…, best_params (per cohort)
And heatmap figures (PNG+SVG) under ``--fig-dir`` (default ``reports/0706/``):
  * ``heatmap_cv_auc``, ``heatmap_<cohort>_auroc``

Example
-------
    python scripts/embedding_grid_eval.py                 # full 10×13 grid
    python scripts/embedding_grid_eval.py --models LR RF --fs ANOVA "Mutual Info"
"""

from __future__ import annotations

import argparse
import json
import tempfile
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Memory
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from hcc_multimodal.eval.grid import (
    FS_ORDER,
    MODEL_ORDER,
    SELECT_K_DEFAULT,
    build_grid_pipeline,
    param_grid,
    positive_scores,
)
from hcc_multimodal.eval.metrics import compute_metrics
from hcc_multimodal.survival.data import load_source_aligned
from hcc_multimodal.train.config import RANDOM_STATE

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="Setting penalty=None will ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

matplotlib.rcParams["svg.fonttype"] = "none"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _labelled(cohort_data):
    """Return (X, y) restricted to patients with a defined rfs_2year label."""
    mask = cohort_data.rfs_2year.notna()
    return cohort_data.X[mask], cohort_data.rfs_2year[mask].astype(int)


# ---------------------------------------------------------------------------
# Nested CV + refit/transfer for one (fs, model) cell
# ---------------------------------------------------------------------------
def nested_cv_auc(fs, model, X, y, select_k, outer_folds, inner_folds, memory):
    outer = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=RANDOM_STATE)
    fold_aucs = []
    for tr, te in outer.split(X, y):
        gs = GridSearchCV(
            build_grid_pipeline(fs, model, select_k, memory=memory),
            param_grid(model),
            cv=StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=RANDOM_STATE),
            scoring="roc_auc",
            refit=True,
            n_jobs=-1,
            error_score="raise",
        )
        gs.fit(X.iloc[tr], y.iloc[tr])
        scores = positive_scores(gs.best_estimator_, X.iloc[te])
        fold_aucs.append(roc_auc_score(y.iloc[te], scores))
    return float(np.mean(fold_aucs)), float(np.std(fold_aucs))


def refit_and_transfer(fs, model, X, y, test_sets, select_k, inner_folds, memory):
    """Fit GridSearchCV on the full resection set, transfer to each test cohort.

    Returns (best_params, {cohort: metrics_dict}).
    """
    gs = GridSearchCV(
        build_grid_pipeline(fs, model, select_k, memory=memory),
        param_grid(model),
        cv=StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=RANDOM_STATE),
        scoring="roc_auc",
        refit=True,
        n_jobs=-1,
        error_score="raise",
    )
    gs.fit(X, y)
    best = gs.best_estimator_

    out = {}
    for cohort, (Xt, yt) in test_sets.items():
        scores = positive_scores(best, Xt)
        out[cohort] = compute_metrics(yt.values, scores)
    return gs.best_params_, out


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------
def draw_heatmap(matrix: pd.DataFrame, title: str, out_stem: Path, cbar_label: str):
    fig, ax = plt.subplots(figsize=(0.95 * matrix.shape[1] + 3, 0.55 * matrix.shape[0] + 2))
    data = matrix.values.astype(float)
    vmin, vmax = np.nanmin(data), np.nanmax(data)
    im = ax.imshow(data, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels(matrix.index, fontsize=9)
    ax.set_xlabel("Feature Selection Technique")
    ax.set_ylabel("Model")
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = data[i, j]
            if np.isnan(v):
                continue
            shade = (v - vmin) / (vmax - vmin + 1e-12)
            color = "white" if shade > 0.7 or shade < 0.15 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color=color)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label(cbar_label)
    fig.tight_layout()
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".png"), dpi=150)
    fig.savefig(out_stem.with_suffix(".svg"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-id", default="9109a6c2")
    p.add_argument("--target", default="rfs_2year")
    p.add_argument("--select-k", type=int, default=SELECT_K_DEFAULT)
    p.add_argument("--outer-folds", type=int, default=5)
    p.add_argument("--inner-folds", type=int, default=3)
    p.add_argument("--cohorts", nargs="+", default=["soramic", "lusanne"])
    p.add_argument("--models", nargs="+", default=MODEL_ORDER)
    p.add_argument("--fs", nargs="+", default=FS_ORDER)
    p.add_argument("--output-dir", type=Path, default=Path("results/eval/grid"))
    p.add_argument("--fig-dir", type=Path, default=Path("reports/0706"))
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Grid eval — model {args.model_id}, select_k={args.select_k}, "
          f"outer={args.outer_folds}, inner={args.inner_folds}")

    # Resection (train) + test cohorts.
    X_res, y_res = _labelled(load_source_aligned(args.model_id, "resection"))
    print(f"  resection labelled: n={len(y_res)} (pos={int(y_res.sum())})")
    test_sets = {}
    for c in args.cohorts:
        Xc, yc = _labelled(load_source_aligned(args.model_id, c))
        test_sets[c] = (Xc, yc)
        print(f"  {c} labelled: n={len(yc)} (pos={int(yc.sum())})")

    cachedir = tempfile.mkdtemp(prefix="grid_cache_")
    memory = Memory(location=cachedir, verbose=0)

    cv_rows, transfer_rows = [], {c: [] for c in args.cohorts}
    for model in args.models:
        for fs in args.fs:
            mean_auc, std_auc = nested_cv_auc(
                fs, model, X_res, y_res, args.select_k,
                args.outer_folds, args.inner_folds, memory,
            )
            best_params, metrics = refit_and_transfer(
                fs, model, X_res, y_res, test_sets, args.select_k, args.inner_folds, memory,
            )
            cv_rows.append({"model": model, "fs": fs,
                            "cv_auc_mean": mean_auc, "cv_auc_std": std_auc})
            for c in args.cohorts:
                transfer_rows[c].append({
                    "model": model, "fs": fs,
                    **metrics[c],
                    "best_params": json.dumps(best_params),
                })
            print(f"  {model:12s} {fs:14s} CV={mean_auc:.3f}±{std_auc:.3f} | "
                  + " | ".join(f"{c}={metrics[c]['auroc']:.3f}" for c in args.cohorts))

    memory.clear(warn=False)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(args.output_dir / "grid_cv_auc.csv", index=False)
    cv_matrix = cv_df.pivot(index="model", columns="fs", values="cv_auc_mean").reindex(
        index=[m for m in MODEL_ORDER if m in args.models],
        columns=[f for f in FS_ORDER if f in args.fs],
    )
    cv_matrix.to_csv(args.output_dir / "grid_cv_auc_matrix.csv")
    draw_heatmap(
        cv_matrix, f"Embedding {args.model_id} — nested {args.outer_folds}-fold CV AUC (resection)",
        args.fig_dir / "heatmap_cv_auc", "CV AUC",
    )

    for c in args.cohorts:
        tdf = pd.DataFrame(transfer_rows[c])
        tdf.to_csv(args.output_dir / f"grid_transfer_{c}.csv", index=False)
        tmat = tdf.pivot(index="model", columns="fs", values="auroc").reindex(
            index=[m for m in MODEL_ORDER if m in args.models],
            columns=[f for f in FS_ORDER if f in args.fs],
        )
        draw_heatmap(
            tmat, f"Embedding {args.model_id} — {c} transfer AUROC",
            args.fig_dir / f"heatmap_{c}_auroc", "AUROC",
        )

    print(f"\nWrote CSVs to {args.output_dir} and figures to {args.fig_dir}")


if __name__ == "__main__":
    main()
