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
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from hcc_multimodal.baselines.config import MODELS as BASELINE_MODELS
from hcc_multimodal.baselines.config import PARAM_GRIDS as BASELINE_PARAM_GRIDS
from hcc_multimodal.eval.grid import (
    FS_ORDER,
    MODEL_ORDER,
    SELECT_K_DEFAULT,
    build_grid_pipeline,
    param_grid,
    positive_scores,
    select_k_grid,
)
from hcc_multimodal.eval.metrics import compute_metrics
from hcc_multimodal.survival.data import MODEL_INPUT, load_source_aligned
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
def nested_cv_auc(fs, model, X, y, select_k_values, outer_folds, inner_folds, memory):
    """Nested-CV AUC for one (fs, model) cell, tuning classifier params *and* the
    number of selected features (``select_k_values``) inside each outer fold."""
    grid = {**param_grid(model), **select_k_grid(fs, select_k_values)}
    outer = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=RANDOM_STATE)
    fold_aucs = []
    for tr, te in outer.split(X, y):
        gs = GridSearchCV(
            build_grid_pipeline(fs, model, select_k_values[0], memory=memory),
            grid,
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


# ---------------------------------------------------------------------------
# Image-only CV rank (§4 of the ablation report)
# ---------------------------------------------------------------------------
def _baseline_zoo() -> dict[str, tuple[object, dict]]:
    """(estimator, param_grid) per classifier, sourced verbatim from ``baselines/config.py``.

    ``LASSO`` = the config ``LR`` (saga solver); its ``PARAM_GRIDS["LR"]`` ``l1_ratio``
    grid is honored directly (sklearn ≥1.8 controls the L1/L2 mix via ``l1_ratio``, not
    ``penalty``). ``RF`` is used as-is.
    """
    return {
        "LASSO": (clone(BASELINE_MODELS["LR"]), BASELINE_PARAM_GRIDS["LR"]),
        "RF": (clone(BASELINE_MODELS["RF"]), BASELINE_PARAM_GRIDS["RF"]),
    }


def nested_cv_auc_estimator(estimator, grid, X, y, outer_folds, inner_folds):
    """Nested stratified-CV AUC for one classifier on all features (no selection).

    Pipeline is ``SimpleImputer(median) → StandardScaler → estimator``; the inner
    ``GridSearchCV`` tunes ``grid`` per outer fold. Returns ``(mean, std)`` over folds.
    """
    outer = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=RANDOM_STATE)
    fold_aucs = []
    for tr, te in outer.split(X, y):
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", clone(estimator)),
        ])
        gs = GridSearchCV(
            pipe, grid,
            cv=StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=RANDOM_STATE),
            scoring="roc_auc", refit=True, n_jobs=-1, error_score="raise",
        )
        gs.fit(X.iloc[tr], y.iloc[tr])
        scores = positive_scores(gs.best_estimator_, X.iloc[te])
        fold_aucs.append(roc_auc_score(y.iloc[te], scores))
    return float(np.mean(fold_aucs)), float(np.std(fold_aucs))


def refit_transfer_estimator(estimator, grid, X, y, test_sets, inner_folds):
    """Refit one classifier (inner GridSearchCV) on all resection, transfer to test cohorts.

    Same no-selection pipeline as ``nested_cv_auc_estimator``. Returns {cohort: auroc}.
    """
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", clone(estimator)),
    ])
    gs = GridSearchCV(
        pipe, grid,
        cv=StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=RANDOM_STATE),
        scoring="roc_auc", refit=True, n_jobs=-1, error_score="raise",
    )
    gs.fit(X, y)
    out = {}
    for cohort, (Xt, yt) in test_sets.items():
        scores = positive_scores(gs.best_estimator_, Xt)
        out[cohort] = compute_metrics(yt.values, scores)["auroc"]
    return out


def refit_and_transfer(fs, model, X, y, test_sets, select_k_values, inner_folds, memory):
    """Fit GridSearchCV on the full resection set, transfer to each test cohort.

    Tunes classifier params *and* the number of selected features. Returns
    (best_params, {cohort: metrics_dict}).
    """
    grid = {**param_grid(model), **select_k_grid(fs, select_k_values)}
    gs = GridSearchCV(
        build_grid_pipeline(fs, model, select_k_values[0], memory=memory),
        grid,
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


def flat_cv_transfer(fs, model, X, y, test_sets, select_k_values, cv, memory):
    """Flat (non-nested) selection: one ``GridSearchCV`` over {classifier params × select_k}.

    ``gs.best_score_`` is the resection CV AUC used to *rank* the cell (mean over ``cv``
    folds; ``std_test_score`` at the best index is its spread), and the same refit
    ``best_estimator_`` transfers to each test cohort. Unlike :func:`nested_cv_auc` there is
    no outer loop — appropriate when the external cohorts (not an outer fold) provide the
    performance estimate and resection CV is used only for selection. Returns
    ``(cv_mean, cv_std, best_params, {cohort: metrics_dict})``.
    """
    grid = {**param_grid(model), **select_k_grid(fs, select_k_values)}
    gs = GridSearchCV(
        build_grid_pipeline(fs, model, select_k_values[0], memory=memory),
        grid, cv=cv, scoring="roc_auc", refit=True, n_jobs=-1, error_score="raise",
    )
    gs.fit(X, y)
    cv_mean = float(gs.best_score_)
    cv_std = float(gs.cv_results_["std_test_score"][gs.best_index_])
    out = {}
    for cohort, (Xt, yt) in test_sets.items():
        out[cohort] = compute_metrics(yt.values, positive_scores(gs.best_estimator_, Xt))
    return cv_mean, cv_std, gs.best_params_, out


def make_cv(n_splits, repeats, seed):
    """Stratified splitter for flat selection: repeated if ``repeats > 1``."""
    if repeats and repeats > 1:
        return RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=repeats, random_state=seed)
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


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
    p.add_argument("--task", choices=["grid", "cv-rank"], default="grid",
                   help="grid: 10×13 classifier×FS grid + transfer on one embedding. "
                        "cv-rank: per-model image-only nested-CV AUC for the ablation report §4.")
    p.add_argument("--model-id", default="9109a6c2",
                   help="grid task: embedding(s) to grid (accepts one or more ids).", nargs="+")
    p.add_argument("--select-k-fracs", type=float, nargs="+", default=None,
                   help="grid task: fraction(s) of features to select, tuned as a "
                        "hyperparameter (e.g. 0.333 0.667). Overrides --select-k.")
    p.add_argument("--model-ids", nargs="+", default=list(MODEL_INPUT),
                   help="cv-rank task: embeddings to rank (default = the 17 ablation models).")
    p.add_argument("--classifiers", nargs="+", default=["LASSO", "RF"],
                   help="cv-rank task: classifiers from baselines/config.py to score.")
    p.add_argument("--target", default="rfs_2year")
    p.add_argument("--select-k", type=int, default=SELECT_K_DEFAULT)
    p.add_argument("--cv-mode", choices=["nested", "flat"], default="nested",
                   help="grid task resection CV: 'nested' = outer/inner nested CV per cell "
                        "(unbiased estimate, §6). 'flat' = single GridSearchCV, rank cells by "
                        "best_score_ (selection only, §7). Use with --cv-repeats.")
    p.add_argument("--cv-repeats", type=int, default=1,
                   help="flat cv-mode: n_repeats for RepeatedStratifiedKFold (1 = plain k-fold).")
    p.add_argument("--seed", type=int, default=RANDOM_STATE,
                   help="flat cv-mode: random_state for the CV split.")
    p.add_argument("--outer-folds", type=int, default=None,
                   help="outer CV folds / flat n_splits (default: grid=5, cv-rank=3).")
    p.add_argument("--inner-folds", type=int, default=3)
    p.add_argument("--cohorts", nargs="+", default=["soramic", "lusanne"])
    p.add_argument("--models", nargs="+", default=MODEL_ORDER)
    p.add_argument("--fs", nargs="+", default=FS_ORDER)
    p.add_argument("--output-dir", type=Path, default=Path("results/eval/grid"))
    p.add_argument("--fig-dir", type=Path, default=Path("reports/0706"))
    return p.parse_args()


def run_cv_rank(args):
    """Per-model image-only nested-CV AUC + Soramic/Lausanne transfer, best of the zoo.

    For each embedding: nested CV AUC on resection for each classifier, plus a refit on
    all resection and transfer AUROC to each cohort. The CV-selected best head (highest
    resection CV AUC) fixes the reported transfer numbers.
    """
    zoo = _baseline_zoo()
    unknown = [c for c in args.classifiers if c not in zoo]
    if unknown:
        raise SystemExit(f"Unknown classifiers {unknown}; available: {list(zoo)}")
    outer = args.outer_folds if args.outer_folds is not None else 3  # §4 default is 3-fold
    print(f"CV-rank — image-only, classifiers={args.classifiers}, "
          f"nested {outer}-fold / inner {args.inner_folds}-fold, all 128 features, "
          f"transfer→{args.cohorts}")

    rows = []
    for model_id in args.model_ids:
        X, y = _labelled(load_source_aligned(model_id, "resection"))
        test_sets = {c: _labelled(load_source_aligned(model_id, c)) for c in args.cohorts}
        per_clf, per_clf_transfer = {}, {}
        for clf in args.classifiers:
            est, grid = zoo[clf]
            per_clf[clf] = nested_cv_auc_estimator(est, grid, X, y, outer, args.inner_folds)
            per_clf_transfer[clf] = refit_transfer_estimator(
                est, grid, X, y, test_sets, args.inner_folds,
            )
        best_head = max(per_clf, key=lambda c: per_clf[c][0])
        best_mean, best_std = per_clf[best_head]
        row = {"model_id": model_id, "best_head": best_head,
               "cv_auc_mean": best_mean, "cv_auc_std": best_std}
        for c in args.cohorts:  # transfer AUROC of the CV-selected head
            row[f"{c}_auroc"] = per_clf_transfer[best_head][c]
        for clf in args.classifiers:  # per-classifier detail
            row[f"{clf.lower()}_mean"], row[f"{clf.lower()}_std"] = per_clf[clf]
            for c in args.cohorts:
                row[f"{clf.lower()}_{c}_auroc"] = per_clf_transfer[clf][c]
        rows.append(row)
        print(f"  {model_id}  best={best_head} {best_mean:.3f}±{best_std:.3f}  | "
              + " ".join(f"{c}={row[f'{c}_auroc']:.3f}" for c in args.cohorts))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values("cv_auc_mean", ascending=False)
    out = args.output_dir / "cv_rank_image_only.csv"
    df.to_csv(out, index=False)
    print(f"\nWrote {out}")


def grid_one_model(model_id, select_k_values, args, out_dir, fig_dir):
    """Full FS × classifier grid on one embedding, with select_k a tuned hyperparameter.

    Writes per-model CSVs/heatmaps under ``out_dir``/``fig_dir`` and returns the
    best-by-CV cell as a dict (model_id, fs, classifier, cv AUC, transfer AUROCs,
    selected feature count).
    """
    X_res, y_res = _labelled(load_source_aligned(model_id, "resection"))
    print(f"  resection labelled: n={len(y_res)} (pos={int(y_res.sum())})")
    test_sets = {}
    for c in args.cohorts:
        Xc, yc = _labelled(load_source_aligned(model_id, c))
        test_sets[c] = (Xc, yc)
        print(f"  {c} labelled: n={len(yc)} (pos={int(yc.sum())})")

    cachedir = tempfile.mkdtemp(prefix="grid_cache_")
    memory = Memory(location=cachedir, verbose=0)

    flat_cv = make_cv(args.outer_folds, args.cv_repeats, args.seed) if args.cv_mode == "flat" else None

    cv_rows, transfer_rows = [], {c: [] for c in args.cohorts}
    cell_transfer = {}  # (model, fs) -> {cohort: auroc}
    cell_params = {}    # (model, fs) -> best_params (incl. selected feature count)
    for model in args.models:
        for fs in args.fs:
            if args.cv_mode == "flat":
                mean_auc, std_auc, best_params, metrics = flat_cv_transfer(
                    fs, model, X_res, y_res, test_sets, select_k_values, flat_cv, memory,
                )
            else:
                mean_auc, std_auc = nested_cv_auc(
                    fs, model, X_res, y_res, select_k_values,
                    args.outer_folds, args.inner_folds, memory,
                )
                best_params, metrics = refit_and_transfer(
                    fs, model, X_res, y_res, test_sets, select_k_values, args.inner_folds, memory,
                )
            cv_rows.append({"model": model, "fs": fs,
                            "cv_auc_mean": mean_auc, "cv_auc_std": std_auc})
            cell_transfer[(model, fs)] = {c: metrics[c]["auroc"] for c in args.cohorts}
            cell_params[(model, fs)] = best_params
            for c in args.cohorts:
                transfer_rows[c].append({
                    "model": model, "fs": fs,
                    **metrics[c],
                    "best_params": json.dumps(best_params),
                })
            print(f"  {model:12s} {fs:14s} CV={mean_auc:.3f}±{std_auc:.3f} | "
                  + " | ".join(f"{c}={metrics[c]['auroc']:.3f}" for c in args.cohorts))

    memory.clear(warn=False)

    out_dir.mkdir(parents=True, exist_ok=True)
    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(out_dir / "grid_cv_auc.csv", index=False)
    cv_matrix = cv_df.pivot(index="model", columns="fs", values="cv_auc_mean").reindex(
        index=[m for m in MODEL_ORDER if m in args.models],
        columns=[f for f in FS_ORDER if f in args.fs],
    )
    cv_matrix.to_csv(out_dir / "grid_cv_auc_matrix.csv")
    if args.cv_mode == "flat":
        proto = (f"repeated {args.outer_folds}×{args.cv_repeats}" if args.cv_repeats > 1
                 else f"{args.outer_folds}-fold")
        cv_title = f"Embedding {model_id} — resection CV AUC ({proto}, seed {args.seed})"
    else:
        cv_title = f"Embedding {model_id} — nested {args.outer_folds}-fold CV AUC (resection)"
    draw_heatmap(cv_matrix, cv_title, fig_dir / "heatmap_cv_auc", "CV AUC")

    for c in args.cohorts:
        tdf = pd.DataFrame(transfer_rows[c])
        tdf.to_csv(out_dir / f"grid_transfer_{c}.csv", index=False)
        tmat = tdf.pivot(index="model", columns="fs", values="auroc").reindex(
            index=[m for m in MODEL_ORDER if m in args.models],
            columns=[f for f in FS_ORDER if f in args.fs],
        )
        draw_heatmap(
            tmat, f"Embedding {model_id} — {c} transfer AUROC",
            fig_dir / f"heatmap_{c}_auroc", "AUROC",
        )

    # Best-by-resection-CV cell across the whole grid, and its transfer.
    best = cv_df.loc[cv_df["cv_auc_mean"].idxmax()]
    key = (best["model"], best["fs"])
    bp = cell_params[key]
    k_param = next((p for p in bp if p.startswith("selector__")), None)
    summary = {
        "model_id": model_id,
        "best_classifier": best["model"],
        "best_fs": best["fs"],
        "cv_auc_mean": float(best["cv_auc_mean"]),
        "cv_auc_std": float(best["cv_auc_std"]),
        "n_features_selected": bp.get(k_param) if k_param else "all",
        "best_params": json.dumps(bp),
    }
    for c in args.cohorts:
        summary[f"{c}_auroc"] = cell_transfer[key][c]
    print(f"  ==> BEST {model_id}: {best['model']}/{best['fs']} "
          f"CV={best['cv_auc_mean']:.3f} k={summary['n_features_selected']} | "
          + " | ".join(f"{c}={summary[f'{c}_auroc']:.3f}" for c in args.cohorts))
    return summary


def run_grid(args):
    if args.outer_folds is None:
        args.outer_folds = 5
    model_ids = args.model_id if isinstance(args.model_id, list) else [args.model_id]

    n_features = load_source_aligned(model_ids[0], "resection").X.shape[1]
    if args.select_k_fracs:
        select_k_values = sorted({max(1, round(f * n_features)) for f in args.select_k_fracs})
    else:
        select_k_values = [args.select_k]
    if args.cv_mode == "flat":
        proto = (f"repeated {args.outer_folds}×{args.cv_repeats}" if args.cv_repeats > 1
                 else f"{args.outer_folds}-fold")
        cv_desc = f"flat select ({proto}, seed {args.seed})"
    else:
        cv_desc = f"nested {args.outer_folds}-fold / inner {args.inner_folds}-fold"
    print(f"Grid eval — models {model_ids}, select_k={select_k_values} "
          f"(tuned; n_features={n_features}), CV={cv_desc}")

    summaries = []
    for model_id in model_ids:
        print(f"\n=== {model_id} ===")
        out_dir = args.output_dir / model_id if len(model_ids) > 1 else args.output_dir
        fig_dir = args.fig_dir / model_id if len(model_ids) > 1 else args.fig_dir
        summaries.append(grid_one_model(model_id, select_k_values, args, out_dir, fig_dir))

    sdf = pd.DataFrame(summaries).sort_values("cv_auc_mean", ascending=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sout = args.output_dir / "grid_best_by_cv.csv"
    sdf.to_csv(sout, index=False)
    print(f"\nWrote per-model best-by-CV summary to {sout}")
    print(sdf.to_string(index=False))


def main():
    args = parse_args()
    if args.task == "cv-rank":
        run_cv_rank(args)
    else:
        run_grid(args)


if __name__ == "__main__":
    main()
