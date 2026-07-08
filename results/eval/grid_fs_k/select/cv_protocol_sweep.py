"""Does the resection-CV-best config transfer to Soramic under a lower-variance CV?

For each CV protocol and each (classifier, FS) cell: one GridSearchCV over
{classifier hyperparams x select_k in [43,85,128]} on resection; gs.best_score_ = the
resection CV AUC of the selected config; refit on all resection -> Soramic/Lausanne.
Across all cells pick the argmax-by-CV cell (what a reviewer deploys) and report transfer.

Writes one CSV per (model, protocol) as soon as it is computed so progress is inspectable.
"""
from __future__ import annotations

import sys
import tempfile
import time
import warnings

import pandas as pd
from joblib import Memory
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from hcc_multimodal.eval.grid import (
    FS_ORDER,
    MODEL_ORDER,
    build_grid_pipeline,
    param_grid,
    positive_scores,
    select_k_grid,
)
from hcc_multimodal.eval.metrics import compute_metrics
from hcc_multimodal.survival.data import load_source_aligned
from hcc_multimodal.train.config import RANDOM_STATE
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="Setting penalty=None will ignore")

import os
SD = os.path.dirname(os.path.abspath(__file__))  # write CSVs next to this script
SELECT_K = [43, 85, 128]  # 1/3, 2/3, ALL features
COHORTS = ["soramic", "lusanne"]

PROTOCOLS = {
    "5fold":  lambda: StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
    "10fold": lambda: StratifiedKFold(10, shuffle=True, random_state=RANDOM_STATE),
    "5x10":   lambda: RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_STATE),
}


def _labelled(cd):
    m = cd.rfs_2year.notna()
    return cd.X[m], cd.rfs_2year[m].astype(int)


def run_model(model_id, models=None, fss=None):
    X, y = _labelled(load_source_aligned(model_id, "resection"))
    tests = {c: _labelled(load_source_aligned(model_id, c)) for c in COHORTS}
    models = models or MODEL_ORDER
    fss = fss or FS_ORDER
    memory = Memory(location=tempfile.mkdtemp(prefix="sweep_"), verbose=0)

    for cvname, mk in PROTOCOLS.items():
        t0 = time.time()
        rows = []
        for clf in models:
            for fs in fss:
                grid = {**param_grid(clf), **select_k_grid(fs, SELECT_K)}
                gs = GridSearchCV(
                    build_grid_pipeline(fs, clf, SELECT_K[0], memory=memory),
                    grid, cv=mk(), scoring="roc_auc", refit=True,
                    n_jobs=-1, error_score="raise",
                )
                gs.fit(X, y)
                kp = next((p for p in gs.best_params_ if p.startswith("selector__")), "")
                row = {"clf": clf, "fs": fs, "cv": float(gs.best_score_),
                       "k": gs.best_params_.get(kp, "all")}
                for c in COHORTS:
                    Xt, yt = tests[c]
                    row[c] = compute_metrics(
                        yt.values, positive_scores(gs.best_estimator_, Xt))["auroc"]
                rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(f"{SD}/sweep_{model_id}_{cvname}.csv", index=False)
        best = df.loc[df["cv"].idxmax()]
        band = df[df["cv"] >= df["cv"].max() - 0.02]
        print(f"[{model_id} {cvname:6s} {time.time()-t0:5.0f}s] "
              f"argmax {best.clf:5s}/{best.fs:13s} k={str(best.k):>3s} CV={best.cv:.3f} "
              f"soramic={best.soramic:.3f} lusanne={best.lusanne:.3f} | "
              f"band(n={len(band)}) soramic {band.soramic.min():.3f}-{band.soramic.max():.3f}",
              flush=True)
    memory.clear(warn=False)


if __name__ == "__main__":
    for mid in (sys.argv[1:] or ["dc7e1d10"]):
        print(f"\n########## {mid} ##########", flush=True)
        run_model(mid)
