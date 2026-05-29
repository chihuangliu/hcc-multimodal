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
import subprocess
from datetime import datetime
from pathlib import Path

import joblib
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import hcc_multimodal
from hcc_multimodal.baselines.data import add_rfs_columns
from hcc_multimodal.contrastive.encoders import ImageEncoder
from hcc_multimodal.contrastive.transform import resample_to_spacing, resampled_shape
from hcc_multimodal.eval.metrics import compute_metrics
from hcc_multimodal.train.config import RANDOM_STATE, SELECT_K
from hcc_multimodal.utils.data import RADIOMICS_FEATURES

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(hcc_multimodal.__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
TRAINING_ROOT = PROJECT_ROOT / "training" / "contrastive"

ABLATION_CLINICAL_CSV = (
    DATA_ROOT / "Ablation" / "Images" / "Clinical" / "2025_Feb_21_RFA_Clinical_Data.csv"
)
ABLATION_RADIOMIC_ROOT = (
    DATA_ROOT / "Ablation" / "Images" / "Radiomics" / "arterial_phase"
)
ABLATION_MRI_ROOT = (
    DATA_ROOT / "Ablation" / "Images" / "Radiomics_nifti_all" / "RFA_images"
)

RESECTION_CLINICAL_CSV = (
    DATA_ROOT
    / "Clinical"
    / "2025_Nov_18_ICL_Resection_Clinical_Outcome_soramic_format.csv"
)
RESECTION_RADIOMIC_CSV = (
    DATA_ROOT / "Resection" / "Images" / "Radiomics" / "radiomic_cluster.csv"
)
RESECTION_MRI_ROOT = DATA_ROOT / "Resection" / "Images" / "Radiomics" / "arterial"

# ---------------------------------------------------------------------------
# Filenames / cache names
# ---------------------------------------------------------------------------
ABLATION_MRI_FILENAME = "MRI_dyn_arterial.nii.gz"
ABLATION_SEG_FILENAME_TEMPLATE = "{sid}_hcc_reg_seg_{lesion_id}.nii.gz"

CONTRASTIVE_METADATA_FILENAME = "metadata.json"
CONTRASTIVE_CHECKPOINT_FILENAME = "best_model.pt"

RESECTION_EMB_CACHE = "resection_img_emb.parquet"
ABLATION_EMB_CACHE_RAW = "ablation_img_emb_raw.parquet"
ABLATION_EMB_CACHE_BBOX = "ablation_img_emb_bbox.parquet"

# downstream classifiers (matches report §3.3 for embedding/concat tasks)
_DOWNSTREAM_MODELS = {
    "lr": LogisticRegression(
        solver="saga",
        penalty="elasticnet",
        l1_ratio=1.0,
        C=1.0,
        max_iter=1000,
        random_state=RANDOM_STATE,
    ),
    "rf": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=PROJECT_ROOT,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _normalize_slice(s: np.ndarray) -> np.ndarray:
    p99 = np.percentile(s, 99)
    s = np.clip(s, 0, p99)
    if p99 > 0:
        s = s / p99
    return s.astype(np.float32)


def _sample_indices(depth: int, n: int | None) -> list[int]:
    if n is None:
        return list(range(depth))
    return np.linspace(0, depth - 1, n, dtype=int).tolist()


def _compute_bbox(
    seg_vol: np.ndarray, pad: int
) -> tuple[np.ndarray, np.ndarray]:
    nz = np.argwhere(seg_vol > 0.5)
    lo = np.maximum(nz.min(axis=0) - pad, 0)
    hi = np.minimum(nz.max(axis=0) + pad, np.array(seg_vol.shape) - 1)
    return lo, hi


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_ablation_outcomes(target: str) -> pd.Series:
    clinical = pd.read_csv(ABLATION_CLINICAL_CSV, encoding="latin-1").dropna(how="all")
    clinical = add_rfs_columns(clinical)
    clinical["SID"] = clinical["SID"].astype(int)
    return clinical.set_index("SID")[target].dropna().astype(int)


def load_resection_outcomes(target: str) -> pd.Series:
    clinical = pd.read_csv(RESECTION_CLINICAL_CSV).dropna(how="all")
    clinical = add_rfs_columns(clinical)
    clinical["SID"] = clinical["SID"].astype(int)
    return clinical.set_index("SID")[target].dropna().astype(int)


def load_ablation_radiomics(
    outcomes: pd.Series, multi_lesion: str
) -> tuple[pd.DataFrame, pd.Series]:
    """Load ablation radiomic features from per-lesion TSV files.

    Returns (X, y) aligned to each other.
    multi_lesion='average'   → one row per patient (features averaged across lesions)
    multi_lesion='per_lesion'→ one row per lesion, outcome replicated from patient
    """
    rows, lesion_ids = [], []
    for lesion_dir in sorted(ABLATION_RADIOMIC_ROOT.iterdir()):
        tsv = lesion_dir / f"{lesion_dir.name}.nii-results.tsv"
        if not tsv.exists():
            continue
        df = pd.read_csv(tsv, sep="\t", nrows=1)
        missing = [f for f in RADIOMICS_FEATURES if f not in df.columns]
        if missing:
            continue
        rows.append(df[RADIOMICS_FEATURES].iloc[0].values)
        lesion_ids.append(lesion_dir.name)

    X_raw = pd.DataFrame(rows, index=lesion_ids, columns=RADIOMICS_FEATURES)

    # derive patient IDs (strip lesion suffix, e.g. "1001007_1" → 1001007)
    patient_ids = X_raw.index.map(lambda s: int(s.rsplit("_", 1)[0]))
    X_raw["_patient_id"] = patient_ids

    # keep only patients with outcomes
    X_raw = X_raw[X_raw["_patient_id"].isin(outcomes.index)]

    if multi_lesion == "average":
        X = X_raw.groupby("_patient_id")[RADIOMICS_FEATURES].mean()
        X.index.name = "SID"
        y = outcomes.loc[X.index]
        return X, y

    # per_lesion: keep lesion-level index, replicate outcome from patient
    X = X_raw.drop(columns=["_patient_id"])
    y = pd.Series(
        [outcomes.loc[int(s.rsplit("_", 1)[0])] for s in X.index],
        index=X.index,
    )
    return X, y


def load_resection_radiomics(outcomes: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(RESECTION_RADIOMIC_CSV).dropna(how="all")
    art_cols = [f"{f}_art" for f in RADIOMICS_FEATURES]
    df = df[["SID"] + art_cols].copy()
    df.columns = ["SID"] + RADIOMICS_FEATURES
    df["SID"] = df["SID"].astype(int)
    df = df.set_index("SID")
    common = df.index.intersection(outcomes.index)
    return df.loc[common], outcomes.loc[common]


# ---------------------------------------------------------------------------
# Image embedding extraction
# ---------------------------------------------------------------------------
class _MRIDataset(Dataset):
    """Flat (patient_id, slice) index for image embedding extraction.

    Supports two modes:
    - Plain (seg_root=None): optionally resample to 1×1×3 mm, then slice.
    - BBox (seg_root given): per-lesion bbox crop from segmentation masks,
      then slice. All lesion crops for a patient share the same pid so the
      caller's mean-pool loop averages them together (Option B).

    Index entries are always (pid, crop_idx, si).  In plain mode crop_idx
    is always -1 and ignored in __getitem__.
    """

    def __init__(
        self,
        patient_ids: list[int],
        mri_root: Path,
        mri_filename_fn,
        n_per_axis: int | None,
        axis: int,
        img_size: int,
        vit_transform,
        resample: bool = False,
        seg_root: Path | None = None,
        bbox_pad: int = 10,
    ):
        self.mri_root = mri_root
        self.mri_filename_fn = mri_filename_fn
        self.axis = axis
        self.resize = transforms.Resize((img_size, img_size), antialias=True)
        self.vit_transform = vit_transform
        self.resample = resample
        self.seg_root = seg_root
        self.bbox_pad = bbox_pad
        self._bbox_mode = seg_root is not None

        # (pid, lo, hi) for each lesion crop; indexed by crop_idx
        self._crops: list[tuple[int, np.ndarray, np.ndarray]] = []
        # unified: (pid, crop_idx, si) — crop_idx=-1 in plain mode
        self._index: list[tuple[int, int, int]] = []

        for pid in patient_ids:
            mri_path = mri_root / str(pid) / mri_filename_fn(pid)
            if not mri_path.exists():
                continue

            if self._bbox_mode:
                lesion_dirs = sorted(seg_root.glob(f"{pid}_*"))
                if not lesion_dirs:
                    continue
                for lesion_dir in lesion_dirs:
                    lesion_id = lesion_dir.name.rsplit("_", 1)[1]
                    seg_path = lesion_dir / ABLATION_SEG_FILENAME_TEMPLATE.format(sid=pid, lesion_id=lesion_id)
                    if not seg_path.exists():
                        continue
                    seg_vol = resample_to_spacing(nib.load(seg_path))
                    if seg_vol.max() < 0.5:
                        continue
                    lo, hi = _compute_bbox(seg_vol, bbox_pad)
                    crop_shape = tuple((hi - lo + 1).tolist())
                    crop_idx = len(self._crops)
                    self._crops.append((pid, lo, hi))
                    for si in _sample_indices(crop_shape[axis], n_per_axis):
                        self._index.append((pid, crop_idx, si))
            else:
                img = nib.load(mri_path)
                shape = resampled_shape(img)[:3] if resample else img.shape[:3]
                for si in _sample_indices(shape[axis], n_per_axis):
                    self._index.append((pid, -1, si))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        pid, crop_idx, si = self._index[i]
        path = self.mri_root / str(pid) / self.mri_filename_fn(pid)

        img = nib.load(path)
        if self.resample or self._bbox_mode:
            vol = resample_to_spacing(img)
        else:
            vol = np.squeeze(np.array(img.dataobj))
            if vol.ndim == 4:
                vol = vol[..., 0]

        if self._bbox_mode:
            _, lo, hi = self._crops[crop_idx]
            hi_safe = np.minimum(hi, np.array(vol.shape[:3]) - 1)
            vol = vol[lo[0]:hi_safe[0]+1, lo[1]:hi_safe[1]+1, lo[2]:hi_safe[2]+1]
            si = min(si, vol.shape[self.axis] - 1)

        s = np.take(vol, si, axis=self.axis)
        t = torch.from_numpy(_normalize_slice(s)).unsqueeze(0)
        t = self.resize(t).repeat(3, 1, 1)
        return self.vit_transform(t), pid


def extract_image_embeddings(
    img_enc: ImageEncoder,
    patient_ids: list[int],
    mri_root: Path,
    mri_filename_fn,
    meta: dict,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    cache_path: Path | None = None,
    resample: bool = False,
    seg_root: Path | None = None,
    bbox_pad: int = 10,
    overwrite_cache: bool = False,
) -> pd.DataFrame:
    """Return (n_patients × embed_dim) DataFrame of mean-pooled image embeddings.

    If cache_path is given and the file exists, load from disk instead of
    re-running the encoder. On a cache miss the result is written to cache_path.

    resample: resample MRI to 1×1×3 mm before slicing (needed for raw ablation MRIs).
    seg_root: if given, use per-lesion bbox crops (bbox mode). All lesion slices
              for a patient are mean-pooled together into one patient embedding.
    """
    if cache_path is not None and cache_path.exists() and not overwrite_cache:
        print(f"  Loading cached embeddings from {cache_path}")
        return pd.read_parquet(cache_path)
    from hcc_multimodal.contrastive.encoders import BACKBONES

    _, weights, _ = BACKBONES[meta["model"]]
    vit_transform = weights.transforms()

    dataset = _MRIDataset(
        patient_ids=patient_ids,
        mri_root=mri_root,
        mri_filename_fn=mri_filename_fn,
        n_per_axis=meta["n_per_axis"],
        axis=meta["axes"] if isinstance(meta["axes"], int) else meta["axes"][0],
        img_size=meta["img_size"],
        vit_transform=vit_transform,
        resample=resample,
        seg_root=seg_root,
        bbox_pad=bbox_pad,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    embed_dim = meta["embed_dim"]
    img_sum: dict[int, np.ndarray] = {}
    img_count: dict[int, int] = {}

    img_enc.eval()
    with torch.no_grad():
        for imgs, pids in loader:
            imgs = imgs.to(device)
            out = img_enc(imgs).cpu().numpy()
            for i, pid in enumerate(pids.tolist()):
                img_sum.setdefault(pid, np.zeros(embed_dim))
                img_count[pid] = img_count.get(pid, 0) + 1
                img_sum[pid] += out[i]

    valid_pids = sorted(img_sum)
    emb = np.stack([img_sum[p] / img_count[p] for p in valid_pids])
    cols = [f"img_{i}" for i in range(embed_dim)]
    df = pd.DataFrame(emb, index=valid_pids, columns=cols)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path)
        print(f"  Cached embeddings → {cache_path}")

    return df


def load_contrastive_model(
    model_id: str, device: torch.device
) -> tuple[ImageEncoder, dict]:
    run_dir = TRAINING_ROOT / model_id
    meta = json.loads((run_dir / CONTRASTIVE_METADATA_FILENAME).read_text())
    img_enc = ImageEncoder(
        meta["model"], meta["embed_dim"], meta["freeze_backbone"]
    ).to(device)
    ckpt = torch.load(
        run_dir / CONTRASTIVE_CHECKPOINT_FILENAME, map_location=device, weights_only=False
    )
    img_enc.load_state_dict(ckpt["img_enc"])
    img_enc.eval()
    return img_enc, meta


# ---------------------------------------------------------------------------
# Downstream pipeline helpers
# ---------------------------------------------------------------------------
def _build_pipeline(model, select_k: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("selector", SelectKBest(f_classif, k=min(select_k, 9999))),
            ("model", clone(model)),
        ]
    )


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
    for name, model in _DOWNSTREAM_MODELS.items():
        k = min(select_k, X_resection_emb.shape[1])
        pipe = _build_pipeline(model, k)
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
    # ablation: align radiomic and embedding indices (patient IDs for average, lesion IDs for per_lesion)
    # for per_lesion, emb is indexed by patient_id; radio is indexed by lesion_id
    # we need to reindex emb to match radio index
    if not X_ablation_radio.index.equals(X_ablation_emb.index):
        # per_lesion: replicate patient embedding for each lesion
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
    for name, model in _DOWNSTREAM_MODELS.items():
        k = min(select_k, X_res.shape[1])
        pipe = _build_pipeline(model, k)
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

    # align ablation patients present in both modalities
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
    for name, model in _DOWNSTREAM_MODELS.items():
        common_res = X_resection_emb.index.intersection(y_resection.index)
        k = min(select_k, X_resection_emb.shape[1])
        emb_pipe = _build_pipeline(model, k)
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

    # --- outcomes ---
    y_ablation_full = load_ablation_outcomes(args.target)
    y_resection_full = load_resection_outcomes(args.target)
    print(f"Ablation outcomes: {len(y_ablation_full)} patients")
    print(f"Resection outcomes: {len(y_resection_full)} patients")

    # --- contrastive model + resection embeddings (loaded once) ---
    resection_emb_df = None
    meta = None
    if needs_contrastive:
        img_enc, meta = load_contrastive_model(args.model_id, device)
        print(f"Loaded contrastive model: {args.model_id}")

        cache_dir = TRAINING_ROOT / args.model_id / "cached_embeddings"
        bbox_mode = meta.get("mri_type") == "raw_bbox"
        abl_cache_name = ABLATION_EMB_CACHE_BBOX if bbox_mode else ABLATION_EMB_CACHE_RAW

        res_pids = [int(p.name) for p in RESECTION_MRI_ROOT.iterdir() if p.is_dir()]
        resection_emb_df = extract_image_embeddings(
            img_enc,
            patient_ids=res_pids,
            mri_root=RESECTION_MRI_ROOT,
            mri_filename_fn=lambda pid: f"{pid}.nii.gz",
            meta=meta,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            cache_path=cache_dir / RESECTION_EMB_CACHE,
            resample=False,  # preprocessed arterial MRIs already at 1×1×3 mm
            overwrite_cache=args.overwrite_cache,
        )
        print(f"Resection embeddings: {resection_emb_df.shape}")

        abl_pids = [int(p.name) for p in ABLATION_MRI_ROOT.iterdir() if p.is_dir()]
        ablation_emb_df = extract_image_embeddings(
            img_enc,
            patient_ids=abl_pids,
            mri_root=ABLATION_MRI_ROOT,
            mri_filename_fn=lambda _: ABLATION_MRI_FILENAME,
            meta=meta,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            cache_path=cache_dir / abl_cache_name,
            resample=True,  # raw ablation MRIs need resampling to 1×1×3 mm
            seg_root=ABLATION_RADIOMIC_ROOT if bbox_mode else None,
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
        key = strategy

        X_abl_radio, y_abl = load_ablation_radiomics(y_ablation_full, strategy)
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

        all_results[key] = strategy_results
        _print_results(strategy, strategy_results)

    # --- save ---
    output = {
        "git_commit": _git_commit(),
        "timestamp": datetime.now().isoformat(),
        "args": vars(args),
        "results": all_results,
    }
    out_path = (
        Path(args.output)
        if args.output
        else (
            PROJECT_ROOT
            / "results"
            / "eval"
            / f"{args.target}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
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
