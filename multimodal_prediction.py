"""Multimodal prediction using contrastive embeddings ± arterial radiomics.

Tasks (--task):
  embeddings - 3-fold CV on contrastive img+gene embeddings only
  concat     - 3-fold CV on embeddings concatenated with arterial radiomics
  radiomics  - 3-fold CV on arterial radiomics only (reproduces reports/0504 baseline)
  ensemble   - train embeddings and radiomics models separately, average probabilities

Feature selection: SelectKBest(f_classif, k=--k) either inside CV (default) or
before CV (--selector_before_cv, diagnostic only — data leakage).

Reproducibility note: --task radiomics with --k 100 --selector_before_cv matches the
SelectKBest F-score (k=100) before-CV result in reports/0504/0504_rfs_baselines_v2.md §3.
"""

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

import hcc_multimodal
from hcc_multimodal.baselines.config import MODELS, MODELS_LR, MODELS_RF, RANDOM_STATE
from hcc_multimodal.baselines.data import add_rfs_columns
from hcc_multimodal.baselines.evaluation import (
    apply_selector_before_cv,
    run_cv_experiment,
)
from hcc_multimodal.baselines.transforms import DataType
from hcc_multimodal.contrastive.config import (
    GENE_SET,
    PREDEFINED_HCC_2Y_CV_GENES,
    RNA_2Y_BEFORE_CV_GENES,
)

_GENE_SETS = {
    "all": GENE_SET,
    "predefined_2y_cv": PREDEFINED_HCC_2Y_CV_GENES,
    "2y_before_cv": RNA_2Y_BEFORE_CV_GENES,
}
from hcc_multimodal.contrastive.data import (
    MRIGeneDataset,
    MRIType,
    _MRI_ROOT_PREPROCESSED,
    _MRI_ROOT_RAW,
    _CLINICAL_PATH,
    _load_gene_matrix,
    _load_outcomes,
)
from hcc_multimodal.contrastive.encoders import BACKBONES, GeneEncoder, ImageEncoder

_DATA_ROOT = Path(hcc_multimodal.__file__).parent.parent / "data"
_RADIOMICS_PATH = _DATA_ROOT / "Radiomics" / "arterial_radiomics.csv"
# "N voxels" is a radiomics feature, not metadata — kept intentionally
_RADIOMICS_META = {"Scan name", "VOI name", "Scan path", "VOI path"}



def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_model(
    model_path: Path, meta: dict, gene_dim: int, device: torch.device
) -> tuple[ImageEncoder, GeneEncoder]:
    img_enc = ImageEncoder(
        meta["model"], meta["embed_dim"], meta["freeze_backbone"]
    ).to(device)
    gene_enc = GeneEncoder(gene_dim, meta["gene_hidden_dim"], meta["embed_dim"]).to(
        device
    )
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    img_enc.load_state_dict(ckpt["img_enc"])
    gene_enc.load_state_dict(ckpt["gene_enc"])
    img_enc.eval()
    gene_enc.eval()
    return img_enc, gene_enc


def extract_embeddings(
    img_enc: ImageEncoder,
    gene_enc: GeneEncoder,
    gene_matrix: pd.DataFrame,
    outcomes: pd.Series,
    meta: dict,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    n_per_axis_override: int | None = None,
) -> pd.DataFrame:
    """Return (patients × 2*embed_dim) DataFrame of mean-pooled img + gene embeddings."""
    _, weights, _ = BACKBONES[meta["model"]]

    mri_type = MRIType(meta.get("mri_type", "preprocessed"))
    mri_root = (
        _MRI_ROOT_PREPROCESSED if mri_type == MRIType.PREPROCESSED else _MRI_ROOT_RAW
    )
    mri_patients = {int(p.name) for p in mri_root.iterdir() if p.is_dir()}
    valid = sorted(mri_patients & set(gene_matrix.index) & set(outcomes.index))

    axes_raw = meta["axes"]
    axes = [axes_raw] if isinstance(axes_raw, int) else list(axes_raw)

    n_per_axis = n_per_axis_override

    dataset = MRIGeneDataset(
        patient_ids=valid,
        gene_matrix=gene_matrix,
        outcomes=outcomes,
        n_per_axis=n_per_axis,
        axes=axes,
        img_size=meta["img_size"],
        transform=weights.transforms(),
        mri_type=mri_type,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    embed_dim = meta["embed_dim"]
    img_sum: dict[int, np.ndarray] = {pid: np.zeros(embed_dim) for pid in valid}
    img_count: dict[int, int] = {pid: 0 for pid in valid}
    gene_emb: dict[int, np.ndarray] = {}

    with torch.no_grad():
        for imgs, genes, _, pids in loader:
            imgs = imgs.to(device)
            genes = genes.to(device)
            img_out = img_enc(imgs).cpu().numpy()
            gene_out = gene_enc(genes).cpu().numpy()
            for i, pid in enumerate(pids.tolist()):
                img_sum[pid] += img_out[i]
                img_count[pid] += 1
                if pid not in gene_emb:
                    gene_emb[pid] = gene_out[i]

    cols = [f"img_emb_{i}" for i in range(embed_dim)] + [
        f"gene_emb_{i}" for i in range(embed_dim)
    ]
    rows = []
    for pid in valid:
        mean_img = img_sum[pid] / max(img_count[pid], 1)
        rows.append(np.concatenate([mean_img, gene_emb[pid]]))

    return pd.DataFrame(np.array(rows), index=valid, columns=cols)


def load_radiomics(patient_ids: set[int]) -> pd.DataFrame:
    """Load arterial radiomics numeric features for the requested patients.

    Used when concatenating with contrastive embeddings: returns only the
    patients that are also in patient_ids (i.e. have embeddings).
    """
    df = pd.read_csv(_RADIOMICS_PATH)
    df["patient_id"] = (
        df["Scan name"].str.replace(r"\.nii\.gz$", "", regex=True).astype(int)
    )
    df = df.set_index("patient_id")
    drop = [c for c in _RADIOMICS_META if c in df.columns]
    df = df.drop(columns=drop).select_dtypes(include=[np.number])
    common = sorted(patient_ids & set(df.index))
    return df.loc[common]


def load_radiomics_baseline(outcome: str) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Load arterial radiomics merged with clinical outcomes, matching the notebook exactly.

    Reproduces the patient set and feature matrix from
    notebooks/baselines/radiomic_arterial_baseline_rfs.ipynb.

    Returns (X, y, x_columns) where X is indexed by SID.
    """
    clinical = add_rfs_columns(pd.read_csv(_CLINICAL_PATH).dropna(how="all"))
    arterial = pd.read_csv(_RADIOMICS_PATH).dropna(how="all")
    arterial["SID"] = (
        arterial["Scan name"].str.replace(r"\.nii\.gz$", "", regex=True).astype(int)
    )
    drop = [c for c in _RADIOMICS_META | {"Scan name"} if c in arterial.columns]
    arterial = arterial.drop(columns=drop)

    merged = arterial.merge(clinical[["SID", outcome]], on="SID")
    merged = merged.set_index("SID").sort_index()
    print(f"Radiomics baseline: {len(merged)} patients matched to clinical outcomes")

    X = merged.drop(columns=[outcome])
    y = merged[outcome].dropna().astype(int)
    x_columns = {c: DataType.CONTINUOUS for c in X.columns}
    return X, y, x_columns


def _cv(
    X: pd.DataFrame,
    y: pd.Series,
    label: str,
    k: int,
    n_splits: int,
    selector_before_cv: bool,
    output_dir: Path,
    models: dict | None = None,
    return_proba: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if models is None:
        models = MODELS
    k_eff = min(k, X.shape[1])
    sel = SelectKBest(f_classif, k=k_eff)
    xcols = {c: DataType.CONTINUOUS for c in X.columns}

    if selector_before_cv:
        save_path = output_dir / f"{label}_selected_features.csv"
        X, y, xcols, cv_kwargs = apply_selector_before_cv(
            X,
            y,
            xcols,
            sel,
            selector_first=False,
            label=label,
            save_path=save_path,
        )
        return run_cv_experiment(
            X,
            y,
            xcols,
            label=label,
            n_splits=n_splits,
            models=models,
            return_proba=return_proba,
            **cv_kwargs,
        )

    return run_cv_experiment(
        X,
        y,
        xcols,
        label=label,
        n_splits=n_splits,
        models=models,
        feature_selector=sel,
        selector_first=False,
        return_proba=return_proba,
    )


def _run_ensemble(
    emb_df: pd.DataFrame,
    radio_df: pd.DataFrame,
    y: pd.Series,
    outcome: str,
    k: int,
    n_splits: int,
    selector_before_cv: bool,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train embeddings and radiomics models on the same folds, then average probabilities.

    Folds are aligned because both calls receive the same y (same patients, same order,
    same RANDOM_STATE), so StratifiedKFold produces identical splits.
    """
    common = sorted(set(emb_df.index) & set(radio_df.index) & set(y.index))
    emb_common = emb_df.loc[common]
    radio_common = radio_df.loc[common]
    y_common = y.loc[common]

    _, emb_folds = _cv(
        emb_common,
        y_common,
        f"emb_{outcome}",
        k,
        n_splits,
        selector_before_cv,
        output_dir,
        return_proba=True,
    )
    _, radio_folds = _cv(
        radio_common,
        y_common,
        f"radio_{outcome}",
        k,
        n_splits,
        selector_before_cv,
        output_dir,
        return_proba=True,
    )

    records, fold_records = [], []
    for model_name in MODELS:
        emb_m = emb_folds[emb_folds["model"] == model_name]
        radio_m = radio_folds[radio_folds["model"] == model_name]
        fold_aucs = []
        for fold_i in range(1, n_splits + 1):
            er = emb_m[emb_m["fold"] == fold_i].iloc[0]
            rr = radio_m[radio_m["fold"] == fold_i].iloc[0]
            assert er["test_indices"] == rr["test_indices"], (
                f"Fold {fold_i} test sets differ between modalities — "
                "ensure both DataFrames have the same patient ordering"
            )
            avg_proba = (np.array(er["test_proba"]) + np.array(rr["test_proba"])) / 2
            y_test = y_common.loc[er["test_indices"]]
            auc = roc_auc_score(y_test, avg_proba)
            fold_aucs.append(auc)
            fold_records.append(
                {
                    "experiment": f"ensemble_{outcome}",
                    "model": model_name,
                    "fold": fold_i,
                    "emb_auc": er["test_auc"],
                    "radio_auc": rr["test_auc"],
                    "ensemble_auc": auc,
                }
            )
        records.append(
            {
                "experiment": f"ensemble_{outcome}",
                "model": model_name,
                "AUC mean": round(np.mean(fold_aucs), 3),
                "AUC std": round(np.std(fold_aucs), 3),
            }
        )

    return pd.DataFrame(records), pd.DataFrame(fold_records)


_TRAINING_ROOT = (
    Path(hcc_multimodal.__file__).parent.parent / "training" / "contrastive"
)
_RESULTS_ROOT = (
    Path(hcc_multimodal.__file__).parent.parent / "results" / "multimodal_prediction"
)


def run(args: argparse.Namespace) -> None:
    if args.task != "radiomics" and args.model_id is None:
        raise SystemExit("model_id is required for tasks: embeddings, concat, ensemble")

    selector_tag = "before_cv" if args.selector_before_cv else "in_cv"
    git_hash = _git_hash()
    if args.task == "radiomics":
        dir_name = f"radiomics_{args.outcome}_{selector_tag}_k{args.k}_{git_hash}"
    else:
        model_meta = json.loads(
            (
                Path(hcc_multimodal.__file__).parent.parent
                / "training"
                / "contrastive"
                / args.model_id
                / "metadata.json"
            ).read_text()
        )
        mri_tag = model_meta.get("mri_type", "preprocessed")
        # None = all slices; int = exactly that many slices
        _n_override = args.n_per_axis
        slice_tag = "nall" if _n_override is None else f"n{_n_override}"
        dir_name = f"{args.task}_{args.model_id}_{args.outcome}_{selector_tag}_k{args.k}_{mri_tag}_{slice_tag}_{git_hash}"

    output_dir = _RESULTS_ROOT / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    meta_cfg: dict = {
        "task": args.task,
        "outcome": args.outcome,
        "model_id": args.model_id,
        "k": args.k,
        "n_splits": args.n_splits,
        "selector_before_cv": args.selector_before_cv,
        "n_per_axis_override": _n_override,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "git_hash": git_hash,
        "timestamp": datetime.now().isoformat(),
    }
    (output_dir / "metadata.json").write_text(json.dumps(meta_cfg, indent=2))

    # --- radiomics baseline (no model needed) ---
    if args.task == "radiomics":
        radio_models = {**MODELS_LR, **MODELS_RF}
        X_r, y_r, _ = load_radiomics_baseline(args.outcome)
        summary_df, fold_df = _cv(
            X_r,
            y_r,
            f"radiomics_{args.outcome}",
            args.k,
            args.n_splits,
            args.selector_before_cv,
            output_dir,
            models=radio_models,
        )
        fold_out = fold_df[["experiment", "model", "fold", "train_auc", "test_auc"]]

    else:
        # --- tasks that need the contrastive model ---
        device = _device()
        print(f"Device: {device}")

        model_path = _TRAINING_ROOT / args.model_id / "best_model.pt"
        meta = json.loads((model_path.parent / "metadata.json").read_text())

        gene_matrix = _load_gene_matrix(_GENE_SETS[meta.get("gene_set", "all")])
        outcomes = _load_outcomes(args.outcome)

        cache_path = (
            _TRAINING_ROOT
            / args.model_id
            / "cached_embeddings"
            / f"emb_{slice_tag}.csv"
        )
        if cache_path.exists():
            print(f"Loading cached embeddings from {cache_path}")
            emb_df = pd.read_csv(cache_path, index_col=0)
        else:
            img_enc, gene_enc = _load_model(
                model_path, meta, gene_matrix.shape[1], device
            )
            print("Extracting contrastive embeddings...")
            emb_df = extract_embeddings(
                img_enc,
                gene_enc,
                gene_matrix,
                outcomes,
                meta,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=device,
                n_per_axis_override=_n_override,
            )
            cache_path.parent.mkdir(exist_ok=True)
            emb_df.to_csv(cache_path)
            print(f"Cached embeddings → {cache_path}")
        print(f"Embedding matrix: {emb_df.shape}")
        y = outcomes.loc[emb_df.index]

        if args.task == "embeddings":
            summary_df, fold_df = _cv(
                emb_df,
                y,
                f"contrastive_emb_{args.outcome}",
                args.k,
                args.n_splits,
                args.selector_before_cv,
                output_dir,
            )
            fold_out = fold_df[["experiment", "model", "fold", "train_auc", "test_auc"]]

        elif args.task == "concat":
            radio_df = load_radiomics(set(emb_df.index))
            common = sorted(set(emb_df.index) & set(radio_df.index))
            combined = pd.concat([emb_df.loc[common], radio_df.loc[common]], axis=1)
            print(
                f"Combined: emb {emb_df.shape[1]} + radiomics {radio_df.shape[1]} = {combined.shape[1]} features"
            )
            summary_df, fold_df = _cv(
                combined,
                y.loc[common],
                f"concat_{args.outcome}",
                args.k,
                args.n_splits,
                args.selector_before_cv,
                output_dir,
            )
            fold_out = fold_df[["experiment", "model", "fold", "train_auc", "test_auc"]]

        elif args.task == "ensemble":
            radio_df = load_radiomics(set(emb_df.index))
            summary_df, fold_df = _run_ensemble(
                emb_df,
                radio_df,
                y,
                args.outcome,
                args.k,
                args.n_splits,
                args.selector_before_cv,
                output_dir,
            )
            fold_out = fold_df[
                ["experiment", "model", "fold", "emb_auc", "radio_auc", "ensemble_auc"]
            ]

    summary_csv = output_dir / "summary.csv"
    fold_csv = output_dir / "folds.csv"
    summary_df.to_csv(summary_csv, index=False)
    fold_out.to_csv(fold_csv, index=False)

    print("\n=== Results ===")
    print(
        summary_df[["experiment", "model", "AUC mean", "AUC std"]].to_string(
            index=False
        )
    )
    print(f"\nSaved → {output_dir}/")


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "model_id",
        nargs="?",
        default=None,
        help=(
            "Run ID from training/contrastive/<model_id>/. "
            "Determines model_path and output_dir. "
            "Not required for --task radiomics."
        ),
    )
    p.add_argument(
        "--outcome",
        default="rfs_2year",
        choices=["rfs_1year", "rfs_2year"],
        help="Prediction target",
    )
    p.add_argument(
        "--task",
        required=True,
        choices=["embeddings", "concat", "radiomics", "ensemble"],
        help="Task to run ('radiomics' reproduces the §3 baseline; others need model_id)",
    )
    p.add_argument(
        "--k",
        type=int,
        default=100,
        help="SelectKBest k (default 100 matches radiomics baseline; capped at feature count)",
    )
    p.add_argument("--n_splits", type=int, default=3, help="CV folds")
    p.add_argument(
        "--selector_before_cv",
        action="store_true",
        help="Apply SelectKBest before CV (label-leaking, diagnostic only)",
    )
    p.add_argument(
        "--n_per_axis",
        type=int,
        default=None,
        help="Number of slices per axis at inference (default: None = all slices). "
        "Pass an integer to fix the slice count, e.g. --n_per_axis 10.",
    )
    p.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for embedding extraction"
    )
    p.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    return p.parse_args()


if __name__ == "__main__":
    run(_parse())
