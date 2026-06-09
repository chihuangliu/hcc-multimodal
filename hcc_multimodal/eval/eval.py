"""External evaluation on the ablation cohort.

Modes (--mode):
  radiomic   – apply the pre-trained joblib radiomic pipeline directly
  embedding  – train LR/RF on resection image embeddings, evaluate on ablation
  concat     – train on [resection radiomics ∥ image embeddings], evaluate on ablation
  ensemble   – average probabilities from radiomic and embedding models
  all        – run all four modes (default)

Multi-lesion strategies for ablation radiomic data (--multi-lesion):
  average    – average radiomic features across lesions per patient (default)
  per_lesion – each lesion is a separate sample; patient-level embedding replicated
  both       – run both strategies and report separately
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import torch

from hcc_multimodal.eval.data import (
    RESECTION_MRI_ROOT,
    RESECTION_EMB_CACHE,
    TRAINING_ROOT,
    extract_image_embeddings,
    get_ablation_config,
    load_ablation_outcomes,
    load_ablation_radiomics,
    load_contrastive_model,
    load_resection_outcomes,
    load_resection_radiomics,
)
from hcc_multimodal.eval.eval_utils import DOWNSTREAM_MODELS, PROJECT_ROOT, build_pipeline
from hcc_multimodal.eval.metrics import compute_metrics
from hcc_multimodal.train.config import SELECT_K
from hcc_multimodal.utils.git import git_commit


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Eval modes
# ---------------------------------------------------------------------------
def eval_radiomic(
    radiomic_model_path: Path,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, dict[str, float]]:
    pipe = joblib.load(radiomic_model_path)
    model_name = radiomic_model_path.stem.split("_")[-1]  # lr or rf
    proba = pipe.predict_proba(X_test)[:, 1]
    return {model_name: compute_metrics(y_test.values, proba)}


def eval_embedding(
    X_resection_emb: pd.DataFrame,
    y_resection: pd.Series,
    X_ablation_emb: pd.DataFrame,
    y_ablation: pd.Series,
    select_k: int,
) -> dict[str, dict[str, float]]:
    common_res = X_resection_emb.index.intersection(y_resection.index)
    common_abl = X_ablation_emb.index.intersection(y_ablation.index)
    results = {}
    for name, model in DOWNSTREAM_MODELS.items():
        k = min(select_k, X_resection_emb.shape[1])
        pipe = build_pipeline(model, k)
        pipe.fit(X_resection_emb.loc[common_res], y_resection.loc[common_res])
        proba = pipe.predict_proba(X_ablation_emb.loc[common_abl])[:, 1]
        results[name] = compute_metrics(y_ablation.loc[common_abl].values, proba)
    return results


def eval_concat(
    X_resection_radio: pd.DataFrame,
    X_resection_emb: pd.DataFrame,
    y_resection: pd.Series,
    X_ablation_radio: pd.DataFrame,
    X_ablation_emb: pd.DataFrame,
    y_ablation: pd.Series,
    select_k: int,
) -> dict[str, dict[str, float]]:
    if not X_ablation_radio.index.equals(X_ablation_emb.index):
        patient_ids = X_ablation_radio.index.map(
            lambda s: int(str(s).rsplit("_", 1)[0]) if "_" in str(s) else int(s)
        )
        abl_emb_aligned = X_ablation_emb.reindex(patient_ids.values)
        abl_emb_aligned.index = X_ablation_radio.index
    else:
        abl_emb_aligned = X_ablation_emb

    res_common = X_resection_radio.index.intersection(
        X_resection_emb.index
    ).intersection(y_resection.index)
    abl_radio_abl_emb_common = X_ablation_radio.index.intersection(
        abl_emb_aligned.index
    ).intersection(y_ablation.index)

    X_res = pd.concat(
        [X_resection_radio.loc[res_common], X_resection_emb.loc[res_common]], axis=1
    )
    X_abl = pd.concat(
        [
            X_ablation_radio.loc[abl_radio_abl_emb_common],
            abl_emb_aligned.loc[abl_radio_abl_emb_common],
        ],
        axis=1,
    )
    y_res = y_resection.loc[res_common]
    y_abl = y_ablation.loc[abl_radio_abl_emb_common]

    results = {}
    for name, model in DOWNSTREAM_MODELS.items():
        k = min(select_k, X_res.shape[1])
        pipe = build_pipeline(model, k)
        pipe.fit(X_res, y_res)
        proba = pipe.predict_proba(X_abl)[:, 1]
        results[name] = compute_metrics(y_abl.values, proba)
    return results


def eval_ensemble(
    X_ablation_radio: pd.DataFrame,
    X_ablation_emb: pd.DataFrame,
    y_ablation: pd.Series,
    radiomic_model_path: Path,
    X_resection_emb: pd.DataFrame,
    y_resection: pd.Series,
    select_k: int,
) -> dict[str, dict[str, float]]:
    radio_pipe = joblib.load(radiomic_model_path)

    if not X_ablation_radio.index.equals(X_ablation_emb.index):
        patient_ids = X_ablation_radio.index.map(
            lambda s: int(str(s).rsplit("_", 1)[0]) if "_" in str(s) else int(s)
        )
        abl_emb_aligned = X_ablation_emb.reindex(patient_ids.values)
        abl_emb_aligned.index = X_ablation_radio.index
    else:
        abl_emb_aligned = X_ablation_emb

    common_abl = X_ablation_radio.index.intersection(
        abl_emb_aligned.index
    ).intersection(y_ablation.index)

    radio_proba = radio_pipe.predict_proba(X_ablation_radio.loc[common_abl])[:, 1]

    results = {}
    for name, model in DOWNSTREAM_MODELS.items():
        common_res = X_resection_emb.index.intersection(y_resection.index)
        k = min(select_k, X_resection_emb.shape[1])
        emb_pipe = build_pipeline(model, k)
        emb_pipe.fit(X_resection_emb.loc[common_res], y_resection.loc[common_res])
        emb_proba = emb_pipe.predict_proba(abl_emb_aligned.loc[common_abl])[:, 1]
        avg_proba = (radio_proba + emb_proba) / 2
        results[name] = compute_metrics(y_ablation.loc[common_abl].values, avg_proba)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(args: argparse.Namespace) -> None:
    needs_contrastive = args.mode in ("embedding", "concat", "ensemble", "all")
    needs_radiomic_model = args.mode in ("radiomic", "concat", "ensemble", "all")

    if needs_contrastive and args.model_id is None:
        raise SystemExit(
            "--model-id is required for modes: embedding, concat, ensemble, all"
        )
    if needs_radiomic_model and args.radiomic_model is None:
        raise SystemExit(
            "--radiomic-model is required for modes: radiomic, concat, ensemble, all"
        )

    multi_lesion_strategies = (
        ["average", "per_lesion"]
        if args.multi_lesion == "both"
        else [args.multi_lesion]
    )

    device = _device()
    print(f"Device: {device}")

    cfg = get_ablation_config(args.ablation_set)

    # --- outcomes ---
    y_ablation_full = load_ablation_outcomes(
        args.ablation_set, args.target, tolerance_months=args.rfs_tolerance_months
    )
    y_resection_full = load_resection_outcomes(
        args.target, tolerance_months=args.rfs_tolerance_months
    )
    print(f"Ablation outcomes ({args.ablation_set}): {len(y_ablation_full)} patients")
    print(f"Resection outcomes: {len(y_resection_full)} patients")

    # --- contrastive model + embeddings (loaded once) ---
    resection_emb_df = None
    ablation_emb_df = None
    meta = None
    if needs_contrastive:
        img_enc, meta = load_contrastive_model(args.model_id, device)
        print(f"Loaded contrastive model: {args.model_id}")

        cache_dir = TRAINING_ROOT / args.model_id / "cached_embeddings"
        bbox_mode = meta.get("mri_type") == "raw_bbox"
        abl_cache_name = f"ablation_{args.ablation_set}_img_emb_{'bbox' if bbox_mode else 'raw'}.parquet"

        res_pids = [int(p.name) for p in RESECTION_MRI_ROOT.iterdir() if p.is_dir()]
        resection_emb_df = extract_image_embeddings(
            img_enc,
            patient_ids=res_pids,
            mri_root=RESECTION_MRI_ROOT,
            pid_to_mri_relpath=lambda pid: f"{pid}/{pid}.nii.gz",
            meta=meta,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            cache_path=cache_dir / RESECTION_EMB_CACHE,
            resample=False,
            overwrite_cache=args.overwrite_cache,
        )
        print(f"Resection embeddings: {resection_emb_df.shape}")

        abl_pids = [int(p.name) for p in cfg.mri_root.iterdir() if p.is_dir()]
        ablation_emb_df = extract_image_embeddings(
            img_enc,
            patient_ids=abl_pids,
            mri_root=cfg.mri_root,
            pid_to_mri_relpath=cfg.pid_to_mri_relpath,
            meta=meta,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            cache_path=cache_dir / abl_cache_name,
            resample=True,
            masks_root=cfg.masks_root if bbox_mode else None,
            pid_to_mask_prefix=cfg.pid_to_mask_prefix if bbox_mode else None,
            bbox_pad=meta.get("bbox_pad", 10),
            overwrite_cache=args.overwrite_cache,
        )
        print(f"Ablation embeddings: {ablation_emb_df.shape}")

    # --- resection radiomic features (loaded once for concat) ---
    resection_radio_df = None
    resection_radio_y = None
    if args.mode in ("concat", "all"):
        resection_radio_df, resection_radio_y = load_resection_radiomics(
            y_resection_full
        )
        print(f"Resection radiomics: {resection_radio_df.shape}")

    all_results: dict = {}

    for strategy in multi_lesion_strategies:
        print(f"\n=== multi-lesion strategy: {strategy} ===")

        X_abl_radio, y_abl = load_ablation_radiomics(
            args.ablation_set, y_ablation_full, strategy
        )
        print(f"Ablation radiomics ({strategy}): {X_abl_radio.shape}")

        strategy_results: dict = {}

        if args.mode in ("radiomic", "all"):
            strategy_results["radiomic"] = eval_radiomic(
                Path(args.radiomic_model), X_abl_radio, y_abl
            )

        if args.mode in ("embedding", "all"):
            strategy_results["embedding"] = eval_embedding(
                resection_emb_df,
                y_resection_full,
                ablation_emb_df,
                y_ablation_full,
                args.select_k,
            )

        if args.mode in ("concat", "all"):
            strategy_results["concat"] = eval_concat(
                resection_radio_df,
                resection_emb_df,
                resection_radio_y,
                X_abl_radio,
                ablation_emb_df,
                y_abl,
                args.select_k,
            )

        if args.mode in ("ensemble", "all"):
            strategy_results["ensemble"] = eval_ensemble(
                X_abl_radio,
                ablation_emb_df,
                y_abl,
                Path(args.radiomic_model),
                resection_emb_df,
                y_resection_full,
                args.select_k,
            )

        all_results[strategy] = strategy_results
        _print_results(strategy, strategy_results)

    # --- save ---
    output = {
        "git_commit": git_commit(),
        "timestamp": datetime.now().isoformat(),
        "args": vars(args),
        "results": all_results,
    }
    if args.output:
        out_path = Path(args.output)
    else:
        parts = [args.mode]
        if args.model_id:
            parts.append(args.model_id)
        parts.append(args.target)
        parts.append(datetime.now().strftime("%Y%m%d_%H%M%S"))
        out_path = (
            PROJECT_ROOT
            / "results"
            / "eval"
            / args.ablation_set
            / f"{'_'.join(parts)}.json"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nSaved → {out_path}")


def _print_results(strategy: str, results: dict) -> None:
    metrics = ["auroc", "auprc", "sensitivity", "specificity", "ppv", "npv", "f1"]
    header = f"{'mode':<12} {'model':<6} " + "  ".join(f"{m:>11}" for m in metrics)
    print(header)
    print("-" * len(header))
    for mode, model_results in results.items():
        for model_name, m in model_results.items():
            row = f"{mode:<12} {model_name:<6} " + "  ".join(
                f"{m.get(k, float('nan')):>11.3f}" for k in metrics
            )
            print(row)


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--ablation-set",
        required=True,
        choices=["lusanne", "soramic"],
        help="Ablation cohort to evaluate on",
    )
    p.add_argument(
        "--model-id",
        default=None,
        help="Contrastive run ID under training/contrastive/",
    )
    p.add_argument(
        "--radiomic-model",
        default=None,
        help="Path to trained joblib radiomic pipeline",
    )
    p.add_argument("--target", default="rfs_2year", choices=["rfs_1year", "rfs_2year"])
    p.add_argument(
        "--mode",
        default="all",
        choices=["radiomic", "embedding", "concat", "ensemble", "all"],
    )
    p.add_argument(
        "--multi-lesion", default="both", choices=["average", "per_lesion", "both"]
    )
    p.add_argument("--select-k", type=int, default=SELECT_K)
    p.add_argument(
        "--rfs-tolerance-months",
        type=int,
        default=0,
        help=(
            "Tolerance window (months) for censored patients near the RFS threshold. "
            "A censored patient last seen at (threshold - tolerance) months is kept as "
            "negative. Default 3: a patient last seen at 21 months with no event is "
            "classified as RFS-2yr negative. Default 0 (no tolerance)."
        ),
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--output", default=None, help="Path to save JSON results")
    p.add_argument(
        "--overwrite-cache",
        action="store_true",
        default=False,
        help="Re-extract embeddings even if a cache file already exists",
    )
    return p.parse_args()


if __name__ == "__main__":
    run(_parse())
