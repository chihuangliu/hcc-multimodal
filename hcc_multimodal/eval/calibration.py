"""Cross-validated Platt scaling calibration on a test cohort.

Loads cached image embeddings, re-fits the downstream LR/RF head on the
resection cohort, then applies 5-fold cross-validated Platt scaling on the
test cohort (soramic or lusanne).

Goal: improve threshold-dependent metrics (sensitivity/specificity/F1 at 0.5)
by aligning probability outputs to the local cohort's positive rate.
AUROC is expected to remain unchanged.

Usage:
  python -m hcc_multimodal.eval.calibration \\
    --ablation-set soramic --model-id dc7e1d10 9109a6c2

  python -m hcc_multimodal.eval.calibration \\
    --ablation-set lusanne --all
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from hcc_multimodal.eval.data import (
    TRAINING_ROOT,
    RESECTION_EMB_CACHE,
    load_ablation_outcomes,
    load_resection_outcomes,
)
from hcc_multimodal.eval.eval_utils import (
    DOWNSTREAM_MODELS,
    PROJECT_ROOT,
    build_pipeline,
)
from hcc_multimodal.eval.metrics import compute_metrics
from hcc_multimodal.train.config import RANDOM_STATE, SELECT_K
from hcc_multimodal.utils.git import git_commit


def _load_embeddings(
    model_id: str, ablation_set: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load cached resection and ablation embeddings for a model.

    Raises FileNotFoundError if either cache is missing (run eval.py first).
    """
    cache_dir = TRAINING_ROOT / model_id / "cached_embeddings"

    res_path = cache_dir / RESECTION_EMB_CACHE
    if not res_path.exists():
        raise FileNotFoundError(
            f"Resection embedding cache not found: {res_path}\n"
            "Run eval.py --mode embedding first to populate the cache."
        )

    meta_path = TRAINING_ROOT / model_id / "metadata.json"
    meta = json.loads(meta_path.read_text())
    bbox_mode = meta.get("mri_type") == "raw_bbox"
    abl_cache_name = (
        f"ablation_{ablation_set}_img_emb_{'bbox' if bbox_mode else 'raw'}.parquet"
    )
    abl_path = cache_dir / abl_cache_name

    if not abl_path.exists():
        raise FileNotFoundError(
            f"Ablation embedding cache not found: {abl_path}\n"
            "Run eval.py --mode embedding --ablation-set {ablation_set} first."
        )

    res_emb = pd.read_parquet(res_path)
    abl_emb = pd.read_parquet(abl_path)
    return res_emb, abl_emb


def _platt_cv(
    y_true: np.ndarray,
    raw_scores: np.ndarray,
    n_folds: int = 5,
    seed: int = RANDOM_STATE,
) -> np.ndarray:
    """5-fold CV Platt scaling: each patient's calibrated prob is from a
    scaler that never saw that patient's label."""
    cal_proba = np.empty_like(raw_scores)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for train_idx, val_idx in skf.split(raw_scores, y_true):
        scaler = LogisticRegression(solver="lbfgs", max_iter=1000)
        scaler.fit(raw_scores[train_idx].reshape(-1, 1), y_true[train_idx])
        cal_proba[val_idx] = scaler.predict_proba(raw_scores[val_idx].reshape(-1, 1))[
            :, 1
        ]
    return cal_proba


def calibrate_model(
    model_id: str,
    ablation_set: str,
    y_resection: pd.Series,
    y_ablation: pd.Series,
    select_k: int,
    n_folds: int,
) -> dict:
    """Run uncalibrated + calibrated evaluation for one model on one cohort."""
    res_emb, abl_emb = _load_embeddings(model_id, ablation_set)

    common_res = res_emb.index.intersection(y_resection.index)
    common_abl = abl_emb.index.intersection(y_ablation.index)

    X_res = res_emb.loc[common_res]
    y_res = y_resection.loc[common_res]
    X_abl = abl_emb.loc[common_abl]
    y_abl = y_ablation.loc[common_abl].values

    results = {}
    for head_name, model in DOWNSTREAM_MODELS.items():
        k = min(select_k, X_res.shape[1])
        pipe = build_pipeline(model, k)
        pipe.fit(X_res, y_res)
        raw_scores = pipe.predict_proba(X_abl)[:, 1]

        cal_scores = _platt_cv(y_abl, raw_scores, n_folds=n_folds)

        results[head_name] = {
            "uncalibrated": compute_metrics(y_abl, raw_scores),
            "calibrated": compute_metrics(y_abl, cal_scores),
        }

    return results


def _discover_model_ids(ablation_set: str) -> list[str]:
    """Return all model IDs that have a cached ablation embedding for this set."""
    found = []
    for run_dir in sorted(TRAINING_ROOT.iterdir()):
        if not run_dir.is_dir():
            continue
        meta_path = run_dir / "metadata.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        bbox_mode = meta.get("mri_type") == "raw_bbox"
        abl_cache = (
            run_dir
            / "cached_embeddings"
            / (
                f"ablation_{ablation_set}_img_emb_{'bbox' if bbox_mode else 'raw'}.parquet"
            )
        )
        if abl_cache.exists():
            found.append(run_dir.name)
    return found


def _print_results(model_id: str, results: dict) -> None:
    metrics = ["auroc", "auprc", "sensitivity", "specificity", "ppv", "npv", "f1"]
    print(f"\n--- {model_id} ---")
    header = f"  {'head':<4} {'stage':<14} " + "  ".join(f"{m:>11}" for m in metrics)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for head_name, stages in results.items():
        for stage, m in stages.items():
            row = f"  {head_name:<4} {stage:<14} " + "  ".join(
                f"{m.get(k, float('nan')):>11.3f}" for k in metrics
            )
            print(row)


def run(args: argparse.Namespace) -> None:
    y_resection = load_resection_outcomes(
        args.target, tolerance_months=args.rfs_tolerance_months
    )
    y_ablation = load_ablation_outcomes(
        args.ablation_set, args.target, tolerance_months=args.rfs_tolerance_months
    )
    print(f"Resection outcomes: {len(y_resection)} patients")
    print(
        f"Ablation outcomes ({args.ablation_set}): {len(y_ablation)} patients  "
        f"(positive rate {y_ablation.mean():.0%})"
    )

    if args.all:
        model_ids = _discover_model_ids(args.ablation_set)
        print(f"Discovered {len(model_ids)} models with cached embeddings.")
    else:
        model_ids = args.model_id

    all_results: dict[str, dict] = {}
    for mid in model_ids:
        print(f"\nCalibrating {mid} ...")
        try:
            result = calibrate_model(
                model_id=mid,
                ablation_set=args.ablation_set,
                y_resection=y_resection,
                y_ablation=y_ablation,
                select_k=args.select_k,
                n_folds=args.n_folds,
            )
            all_results[mid] = result
            _print_results(mid, result)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")

    output = {
        "git_commit": git_commit(),
        "timestamp": datetime.now().isoformat(),
        "args": vars(args),
        "results": all_results,
    }

    if args.output:
        out_path = Path(args.output)
    else:
        tag = "all" if args.all else "_".join(model_ids)
        out_path = (
            PROJECT_ROOT
            / "results"
            / "eval"
            / args.ablation_set
            / f"calibration_{tag}_{args.target}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nSaved → {out_path}")


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--ablation-set",
        required=True,
        choices=["lusanne", "soramic"],
    )
    p.add_argument(
        "--model-id",
        nargs="+",
        default=[],
        metavar="ID",
        help="One or more contrastive model IDs (ignored if --all is set)",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Run all models that have a cached ablation embedding",
    )
    p.add_argument("--target", default="rfs_2year", choices=["rfs_1year", "rfs_2year"])
    p.add_argument("--n-folds", type=int, default=5, help="CV folds for Platt scaling")
    p.add_argument("--select-k", type=int, default=SELECT_K)
    p.add_argument(
        "--rfs-tolerance-months",
        type=int,
        default=0,
    )
    p.add_argument("--output", default=None, help="Path to save JSON results")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    if not args.all and not args.model_id:
        raise SystemExit("Provide --model-id ID [ID ...] or --all")
    run(args)
