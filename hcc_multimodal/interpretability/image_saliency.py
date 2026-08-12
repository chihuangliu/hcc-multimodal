"""Image-branch saliency for the deployed 2-year RFS model ensemble.

The image counterpart of ``mechanistic_gene_attribution``: instead of decomposing the
downstream decision back to *genes* through the shared 128-dim space, it decomposes it
back to *MRI voxels* through the frozen image encoder.

Why the decomposition is exact
------------------------------
A patient's embedding is the **mean over every sagittal slice** of their volume
(``eval/data.py:extract_image_embeddings``), and each ensemble member is linear, so with
``z̄ = (1/S) Σ_s f(x_s)`` and

    S(z) = (1/M) Σ_m σ(a_m(β_m·z + b_m) + c_m)

the local decision direction ``β_eff = ∇_z logit S(z̄)`` gives a per-slice contribution

    c_s = β_eff · f(x_s) / S,      Σ_s c_s = β_eff · z̄   (exact)

so *which* slices drive the score is read off the model rather than chosen by hand. Note
``c_s`` decomposes the **linearised** logit exactly, not ``logit S`` itself — ``β_eff`` is
the first-order term of a mean-of-sigmoids at that patient's operating point.

Stages
------
0. Select exemplar patients per cohort by outcome × prediction (see ``CASES``), plus
   ``--extra-tp``/``--extra-tn`` further confident hits. ``--resection-head`` chooses
   whether resection is scored in-sample by the whole-cohort head (``full``, the
   default) or out-of-fold (``oof``); Soramic always uses the whole-cohort head.
   ``--pin-case cohort:label=SID`` adds exemplars chosen on some other criterion — see
   ``--screen`` below, which ranks every confident hit by how close its extreme slices
   land to the liver so a legible exemplar can be pinned without hand-editing anything.
1. ``β_eff`` per patient, from the head that produced that patient's probability.
2. Replicate the deployed slice pipeline by instantiating the very same
   ``eval/data.py:_MRIDataset`` the embedding extraction uses, then **gate** on
   reproducing the cached ``z̄``.
3. ``c_s`` for every slice.
4. Gradient×Input over every slice, mapped back to the native voxel grid and stacked
   into a 3D saliency volume (NIfTI).
5. Integrated Gradients on the top-|c_s| slices, with a completeness residual.

Degenerate slices
-----------------
``_normalize_slice``'s ``np.clip(s, 0, p99)`` inverts its bounds when a slice's 99th
percentile is negative, so background-only slices of a negative-background volume reach
the encoder as a *constant* image far outside the trained range — and still enter the
mean-pooled embedding. Those slices are flagged, excluded from the per-voxel figures
(a constant image has no spatial story), and their share of ``Σ|c_s|`` is reported.

The model input grid is **not** the slice grid: ``BACKBONE_TRANSFORMS['vit_b_32']``
resizes the 224 slice to 256 and centre-crops back to 224, so the model never sees a
~6% border. ``map_to_native`` inverts that (zero border, then resize), which keeps the
attribution honest about the region the encoder could not respond to.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hcc_multimodal.contrastive.encoders import BACKBONE_TRANSFORMS
from hcc_multimodal.contrastive.transform import resample_to_spacing
from hcc_multimodal.eval.data import (
    RESECTION_MRI_ROOT,
    _MRIDataset,
    get_ablation_config,
    load_contrastive_model,
)
from hcc_multimodal.eval.grid import positive_scores
from hcc_multimodal.interpretability.mechanistic_gene_attribution import (
    _make_ensemble,
    beta_effective,
    ensemble_score,
    load_members,
    unwind_ensemble,
)
from hcc_multimodal.survival.data import load_embeddings, load_survival_outcomes
from hcc_multimodal.train.config import RANDOM_STATE

# Case label -> human-readable description. Order is the figure panel order.
CASES = {
    "tp_high": "y=1, pred=1, high p (confident hit)",
    "tn_low": "y=0, pred=0, low p (confident hit)",
    "fp_high": "y=0, pred=1, high p (confident miss)",
    "fp_borderline": "y=0, pred=1, p near 0.5",
    "fn_borderline": "y=1, pred=0, p near 0.5",
    "fn_low": "y=1, pred=0, low p (confident miss)",
}

N_SPLITS = 3


# Pinned-case labels (``--pin-case``): exemplars chosen by the screening pass rather than
# by probability rank, so they must never be pooled with the rank-selected cases.
PINNED_DESC = {
    "tp_liver": "y=1, pred=1 — extremes nearest the liver (screened, not rank-selected)",
    "tn_liver": "y=0, pred=0 — extremes nearest the liver (screened, not rank-selected)",
}


def case_desc(case: str) -> str:
    """Description for a base case, one of its ``_2``/``_3``... rank extras, or a pin."""
    base, _, rank = case.rpartition("_")
    if rank.isdigit() and base in CASES:
        return f"{CASES[base]}, rank {rank}"
    if case in CASES:
        return CASES[case]
    return PINNED_DESC.get(case, case)


# ---------------------------------------------------------------------------
# Stage 0: heads + case selection
# ---------------------------------------------------------------------------
def fit_heads(model_id: str, specs: list[dict], target: str):
    """Fit the frozen model ensemble on resection embeddings.

    Returns ``(X, y, oof, fold, fold_heads, full_head)``. ``oof`` is the out-of-fold
    positive-class score and ``fold`` records which fold held each patient out, so the
    same held-out head can be reused for that patient's attribution.
    """
    X = load_embeddings(model_id, "resection")
    y = load_survival_outcomes("resection")[target].dropna().astype(int)
    common = X.index.intersection(y.index)
    X, y = X.loc[common], y.loc[common]

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof = pd.Series(index=X.index, dtype=float)
    fold = pd.Series(index=X.index, dtype=int)
    fold_heads = {}
    for k, (tr, va) in enumerate(skf.split(X, y)):
        est = _make_ensemble(specs).fit(X.iloc[tr], y.iloc[tr])
        oof.iloc[va] = positive_scores(est, X.iloc[va])
        fold.iloc[va] = k
        fold_heads[k] = est
    full_head = _make_ensemble(specs).fit(X, y)
    return X, y, oof, fold, fold_heads, full_head


def pick_cases(p: pd.Series, y: pd.Series,
               extra_tp: int = 0, extra_tn: int = 0,
               pinned: dict[str, int] | None = None) -> dict[str, int | None]:
    """Deterministic exemplar per case label; ``None`` when a case is unpopulated.

    ``extra_tp``/``extra_tn`` append the next-ranked confident hits (``tp_high_2``,
    ``tp_high_3``, ...) so the figures can show more than a single example of the model
    getting a patient right — one exemplar cannot distinguish a real pattern from a
    coincidence.

    ``pinned`` appends ``label -> SID`` rows chosen outside this ranking (``--pin-case``),
    for exemplars selected on some other criterion — e.g. the screening pass picking the
    confident hits whose extreme slices land nearest the liver. They are kept under their
    own labels so nothing downstream mistakes them for rank-selected cases.
    """
    d = pd.DataFrame({"p": p, "y": y})
    picks: dict[str, int | None] = {}
    picks["tp_high"] = d[d.y == 1].p.idxmax()
    picks["tn_low"] = d[d.y == 0].p.idxmin()
    picks["fp_high"] = d[d.y == 0].p.idxmax()
    fp = d[(d.y == 0) & (d.p > 0.5)]
    picks["fp_borderline"] = (fp.p - 0.5).abs().idxmin() if len(fp) else None
    fn = d[(d.y == 1) & (d.p < 0.5)]
    picks["fn_borderline"] = (fn.p - 0.5).abs().idxmin() if len(fn) else None
    picks["fn_low"] = d[d.y == 1].p.idxmin()

    taken = {s for s in picks.values() if s is not None}
    for label, sub, ascending, n in (
        ("tp_high", d[d.y == 1], False, extra_tp),
        ("tn_low", d[d.y == 0], True, extra_tn),
    ):
        ranked = sub.sort_values("p", ascending=ascending).index.tolist()
        for r in range(1, n + 1):
            sid = ranked[r] if r < len(ranked) else None
            picks[f"{label}_{r + 1}"] = None if sid in taken else sid
            taken.add(sid)

    for label, sid in (pinned or {}).items():
        picks[label] = int(sid) if sid in p.index else None
        if picks[label] is None:
            print(f"  ! pinned {label}={sid}: not in this cohort, skipped")
    return picks


def build_case_table(model_id, specs, target, cohorts, resection_head: str = "full",
                     extra_tp: int = 0, extra_tn: int = 0,
                     pinned: dict[str, dict[str, int]] | None = None,
                     ) -> tuple[pd.DataFrame, dict]:
    """One row per (cohort, case) with SID, label, probability and attributing head.

    ``fold`` is the index of the held-out head to attribute that patient with, or ``-1``
    for the whole-cohort head. With ``resection_head='full'`` every row is ``-1``: the
    resection cohort is scored and attributed by the head fit on all of it, so the
    reported ``p`` is in-sample and the same head is behind every panel.
    """
    X_res, y_res, oof, fold, fold_heads, full_head = fit_heads(model_id, specs, target)
    p_res_full = pd.Series(positive_scores(full_head, X_res), index=X_res.index)

    rows = []
    scores = {}
    for cohort in cohorts:
        if cohort == "resection":
            p = oof if resection_head == "oof" else p_res_full
            y = y_res
        else:
            Xc = load_embeddings(model_id, cohort)
            yc = load_survival_outcomes(cohort)[target].dropna().astype(int)
            common = Xc.index.intersection(yc.index)
            Xc, yc = Xc.loc[common], yc.loc[common]
            p = pd.Series(positive_scores(full_head, Xc), index=Xc.index)
            y = yc
        scores[cohort] = (p, y)
        picks = pick_cases(p, y, extra_tp, extra_tn, (pinned or {}).get(cohort))
        for order, (case, sid) in enumerate(picks.items()):
            if sid is None:
                print(f"  ! {cohort}/{case}: no patient in this category, skipped")
                continue
            rows.append({
                "cohort": cohort,
                "case": case,
                "case_desc": case_desc(case),
                "case_order": order,
                "SID": int(sid),
                "y": int(y[sid]),
                "p": float(p[sid]),
                "fold": int(fold[sid])
                        if (cohort == "resection" and resection_head == "oof") else -1,
            })
    heads = {"fold_heads": fold_heads, "full_head": full_head,
             "X_res": X_res, "y_res": y_res, "scores": scores}
    return pd.DataFrame(rows), heads


# ---------------------------------------------------------------------------
# Stage 1: per-patient decision direction
# ---------------------------------------------------------------------------
def patient_beta(head, z_bar: np.ndarray, embed_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """``β_eff = ∇_z logit S`` at this patient's own embedding, plus member shares."""
    params = unwind_ensemble(head, embed_dim)
    return beta_effective(params, z_bar)


# ---------------------------------------------------------------------------
# Stage 2: the deployed slice pipeline
# ---------------------------------------------------------------------------
def _cohort_source(cohort: str):
    """``(mri_root, pid_to_relpath, resample)`` exactly as ``eval.eval`` extracts them."""
    if cohort == "resection":
        return RESECTION_MRI_ROOT, (lambda pid: f"{pid}/{pid}.nii.gz"), False
    cfg = get_ablation_config(cohort)
    return cfg.mri_root, cfg.pid_to_mri_relpath, True


def build_slice_dataset(cohort: str, pid: int, meta: dict) -> _MRIDataset:
    """One patient's slices, byte-identical to what the embedding cache was built from."""
    mri_root, relpath, resample = _cohort_source(cohort)
    axis = meta["axes"] if isinstance(meta["axes"], int) else meta["axes"][0]
    return _MRIDataset(
        patient_ids=[pid],
        mri_root=mri_root,
        pid_to_mri_relpath=relpath,
        n_per_axis=meta["n_per_axis"],
        axis=axis,
        img_size=meta["img_size"],
        vit_transform=BACKBONE_TRANSFORMS[meta["model"]](),
        resample=resample,
    )


def load_mask(cohort: str, pid: int) -> np.ndarray | None:
    """Tumour mask on the same voxel grid as the sliced volume, or ``None``."""
    if cohort == "resection":
        path = RESECTION_MRI_ROOT / str(pid) / f"{pid}_hcc_seg.nii.gz"
        if not path.exists():
            return None
        return (np.squeeze(np.asarray(nib.load(path).dataobj)) > 0.5).astype(np.uint8)

    cfg = get_ablation_config(cohort)
    files = sorted(cfg.masks_root.glob(f"{cfg.pid_to_mask_prefix(pid)}_hcc_seg_reg_*.nii.gz"))
    if not files:
        return None
    arrs = [resample_to_spacing(nib.load(f)) for f in files]
    seg = np.maximum.reduce(arrs) if len(arrs) > 1 else arrs[0]
    return (seg > 0.5).astype(np.uint8)


def degenerate_flags(vol: np.ndarray, axis: int, slice_ids) -> np.ndarray:
    """Slices that reach the encoder as a **constant** image.

    Flagged by the mechanism itself rather than by testing the tensor:
    ``_normalize_slice``'s ``np.clip(s, 0, p99)`` collapses the slice to a constant exactly
    when ``p99 <= 0`` (``a_min > a_max`` makes numpy return ``p99`` everywhere, and the
    ``if p99 > 0`` rescale is then skipped). Testing the tensor for constancy does not
    work — two bilinear resizes leave ~1e-5 of float32 noise on a constant at ~-12, and the
    ImageNet normalisation then gives each channel its own constant, so neither an absolute
    nor a whole-tensor std threshold is reliable.
    """
    return np.array([
        np.percentile(np.take(vol, si, axis=axis), 99) <= 0 for si in slice_ids
    ], dtype=bool)


def slice_foreground_fraction(vol: np.ndarray, axis: int, slice_ids,
                              rel_thresh: float = 0.10) -> np.ndarray:
    """Fraction of each slice's voxels above ``rel_thresh × p99(volume)``.

    The reference percentile is taken over the **whole volume**, not per slice: the
    per-slice p99 that ``_normalize_slice`` uses is exactly what degenerates on
    background-only slices, so it cannot distinguish "no anatomy" from "faint anatomy".
    A lateral body-wall or air slice scores near zero here; a mid-abdomen slice does not.
    """
    ref = float(np.percentile(vol, 99))
    thr = rel_thresh * ref if ref > 0 else 0.0
    return np.array([float((np.take(vol, si, axis=axis) > thr).mean())
                     for si in slice_ids], dtype=float)


def tumour_slice_distance(tumour_per_slice: np.ndarray | None) -> np.ndarray | None:
    """Distance, in slice indices, from each slice to the nearest tumour-bearing slice.

    ``0`` inside the tumour's slice extent. The tumour mask is the only anatomical anchor
    in this dataset — there is no liver or body segmentation — so proximity to it is the
    available proxy for "this slice is at the liver". The slice need not contain tumour.
    """
    if tumour_per_slice is None:
        return None
    hit = np.flatnonzero(tumour_per_slice > 0)
    if hit.size == 0:
        return None
    idx = np.arange(len(tumour_per_slice))
    return np.abs(idx[:, None] - hit[None, :]).min(axis=1).astype(float)


def tumour_profile(cohort: str, pid: int, dataset: _MRIDataset,
                   slice_ids: np.ndarray) -> np.ndarray | None:
    """Tumour voxels per sliced slice, or ``None`` when no usable mask exists."""
    mask = load_mask(cohort, pid)
    if mask is None:
        return None
    if mask.shape != dataset._vols[pid].shape:
        print(f"  ! mask shape {mask.shape} != volume {dataset._vols[pid].shape}, ignored")
        return None
    per_slice = mask.sum(axis=tuple(i for i in range(3) if i != dataset.axis))
    return per_slice[slice_ids]


def slice_spacing_mm(cohort: str, pid: int, axis: int) -> float:
    """Spacing along the slicing axis of the *sliced* volume, in mm."""
    affine = volume_affine(cohort, pid)
    return float(np.linalg.norm(affine[:3, axis]))


def volume_affine(cohort: str, pid: int) -> np.ndarray:
    """Affine of the *sliced* volume — rescaled when the loader resampled it."""
    mri_root, relpath, resample = _cohort_source(cohort)
    img = nib.load(mri_root / relpath(pid))
    affine = img.affine.copy()
    if not resample:
        return affine
    native = np.array(img.header.get_zooms()[:3], dtype=float)
    target = np.array([1.0, 1.0, 3.0])
    return affine @ np.diag(np.concatenate([target / native, [1.0]]))


# ---------------------------------------------------------------------------
# Attribution grid <-> native slice grid
# ---------------------------------------------------------------------------
def map_to_native(a: np.ndarray, native_hw: tuple[int, int],
                  resize_size: int = 256, crop: int = 224) -> np.ndarray:
    """Undo ``Resize(resize_size) → CenterCrop(crop) → Resize(224)`` for an attribution map.

    The centre crop means the model never saw a border of the slice, so that border gets
    exactly zero attribution rather than an extrapolated value. The resized map is
    rescaled to preserve its **absolute** mass, so a slice's attribution keeps the
    magnitude it had on the model's own grid.

    Normalising on the *signed* sum instead is unusable here: an attribution map carries
    cancelling positive and negative mass, so ``out.sum()`` is a small difference of large
    numbers and the interpolation's own error on it is unbounded in relative terms. On
    ``soramic/1905004`` slice 332 it collapsed to 2.4e-08 against a signed sum of 1.2e-03,
    scaling that slice by 4.9e+04 until it carried 97% of the patient's ``Σ|attr|`` and
    set the colour scale of the whole MIP; a neighbouring slice landed on the opposite
    sign and had its map inverted. The absolute mass cannot cancel, so the ratio below
    stays at the interpolation's area factor. The signed sum is then preserved only to the
    interpolation's accuracy — which is all this map needs, as the exact chain that
    ``c_s`` belongs to never passes through here.
    """
    pad = (resize_size - crop) // 2
    canvas = np.zeros((resize_size, resize_size), dtype=np.float32)
    canvas[pad:pad + crop, pad:pad + crop] = a
    out = F.interpolate(
        torch.from_numpy(canvas)[None, None], size=native_hw,
        mode="bilinear", align_corners=False, antialias=True,
    )[0, 0].numpy()
    m0, m1 = float(np.abs(canvas).sum()), float(np.abs(out).sum())
    if m1 > 0:
        out = out * (m0 / m1)
    return out


# ---------------------------------------------------------------------------
# Stages 3-4: c_s and the Gradient x Input volume
# ---------------------------------------------------------------------------
def _target(img_enc, x: torch.Tensor, beta: torch.Tensor, n_slices: int) -> torch.Tensor:
    """Per-slice contribution ``β_eff · f(x_s) / S`` — the quantity we attribute."""
    return (img_enc(x) @ beta) / n_slices


def slice_contributions_and_gradxinput(
    img_enc, dataset: _MRIDataset, pid: int, beta: torch.Tensor,
    device: torch.device, batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(c_s, saliency_volume, z_bar, slice_ids, degenerate)`` for one patient.

    ``saliency_volume`` is Gradient×Input mapped back to the native voxel grid and
    stacked along the slicing axis, so it is co-registered with the source volume.

    ``degenerate`` (see :func:`degenerate_flags`) marks slices that reach the encoder as a
    constant image. They carry no anatomy but do enter the mean-pooled embedding, so they
    are tracked rather than silently dropped.
    """
    axis = dataset.axis
    vol = dataset._vols[pid]
    native_hw = tuple(np.delete(np.array(vol.shape), axis).tolist())
    slice_ids = [si for p, si in dataset._index if p == pid]
    n_slices = len(dataset._index)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    c_s = np.zeros(n_slices, dtype=np.float64)
    emb_sum = np.zeros(len(beta), dtype=np.float64)
    maps = np.zeros((n_slices,) + tuple(native_hw), dtype=np.float32)

    degenerate = degenerate_flags(vol, axis, slice_ids)

    cursor = 0
    for imgs, _pids in loader:
        x = imgs.to(device).requires_grad_(True)
        emb = img_enc(x)
        t = (emb @ beta) / n_slices
        grad, = torch.autograd.grad(t.sum(), x)
        attr = (x * grad).sum(dim=1).detach().cpu().numpy()   # (B, 224, 224)

        c_s[cursor:cursor + len(x)] = t.detach().cpu().numpy()
        emb_sum += emb.detach().cpu().numpy().sum(axis=0)
        for j in range(len(x)):
            maps[cursor + j] = map_to_native(attr[j], native_hw)
        cursor += len(x)

    z_bar = emb_sum / n_slices
    # Re-embed the maps into the native volume shape along the slicing axis.
    saliency = np.moveaxis(maps, 0, axis).astype(np.float32)
    return c_s, saliency, z_bar, np.array(slice_ids), degenerate


def slice_contributions_only(
    img_enc, dataset: _MRIDataset, pid: int, beta: torch.Tensor,
    device: torch.device, batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Forward-only twin of :func:`slice_contributions_and_gradxinput`.

    Returns ``(c_s, z_bar, slice_ids, degenerate)`` — identical values, but without the
    per-slice backward pass and the native-grid remapping. That is roughly a tenth of the
    cost of the full per-patient pipeline (whose time is dominated by Integrated
    Gradients), which is what makes screening the whole cohort affordable.
    """
    axis = dataset.axis
    vol = dataset._vols[pid]
    slice_ids = [si for p, si in dataset._index if p == pid]
    n_slices = len(dataset._index)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    c_s = np.zeros(n_slices, dtype=np.float64)
    emb_sum = np.zeros(len(beta), dtype=np.float64)

    cursor = 0
    with torch.no_grad():
        for imgs, _pids in loader:
            emb = img_enc(imgs.to(device))
            c_s[cursor:cursor + len(emb)] = ((emb @ beta) / n_slices).cpu().numpy()
            emb_sum += emb.cpu().numpy().sum(axis=0)
            cursor += len(emb)

    z_bar = emb_sum / n_slices
    return c_s, z_bar, np.array(slice_ids), degenerate_flags(vol, axis, slice_ids)


# ---------------------------------------------------------------------------
# Stage 5: Integrated Gradients on the top-|c_s| slices
# ---------------------------------------------------------------------------
def _baseline_tensor(kind: str, x: torch.Tensor, meta: dict, device) -> torch.Tensor:
    if kind == "zero":
        vit = BACKBONE_TRANSFORMS[meta["model"]]()
        black = vit(torch.zeros(3, meta["img_size"], meta["img_size"]))
        return black.to(device).unsqueeze(0).expand_as(x).contiguous()
    if kind == "blur":
        from torchvision.transforms.v2.functional import gaussian_blur
        return gaussian_blur(x, kernel_size=[31, 31], sigma=[10.0, 10.0])
    raise ValueError(f"unknown baseline {kind!r}")


def integrated_gradients_slices(
    img_enc, dataset: _MRIDataset, pid: int, beta: torch.Tensor, slice_rows: list[int],
    meta: dict, device: torch.device, steps: int, baseline: str, batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Midpoint-rule IG on selected slices.

    Returns ``(ig_native, delta, residual)`` where ``residual = Σ_pixels IG − Δtarget``
    is the completeness check (expected ~0 up to the quadrature error).
    """
    axis = dataset.axis
    vol = dataset._vols[pid]
    native_hw = tuple(np.delete(np.array(vol.shape), axis).tolist())
    n_slices = len(dataset._index)

    ig_native = np.zeros((len(slice_rows),) + tuple(native_hw), dtype=np.float32)
    delta = np.zeros(len(slice_rows))
    residual = np.zeros(len(slice_rows))

    alphas = (torch.arange(steps, dtype=torch.float32, device=device) + 0.5) / steps
    for start in range(0, len(slice_rows), batch_size):
        rows = slice_rows[start:start + batch_size]
        x = torch.stack([dataset[r][0] for r in rows]).to(device)
        base = _baseline_tensor(baseline, x, meta, device)

        total = torch.zeros_like(x)
        for a in alphas:
            xi = (base + a * (x - base)).clone().requires_grad_(True)
            t = _target(img_enc, xi, beta, n_slices)
            g, = torch.autograd.grad(t.sum(), xi)
            total += g
        ig = (x - base) * (total / steps)

        with torch.no_grad():
            d = (_target(img_enc, x, beta, n_slices)
                 - _target(img_enc, base, beta, n_slices)).cpu().numpy()

        ig_sum = ig.sum(dim=(1, 2, 3)).cpu().numpy()
        attr = ig.sum(dim=1).cpu().numpy()
        for j in range(len(rows)):
            ig_native[start + j] = map_to_native(attr[j], native_hw)
        delta[start:start + len(rows)] = d
        residual[start:start + len(rows)] = ig_sum - d
    return ig_native, delta, residual


# ---------------------------------------------------------------------------
# Per-patient driver
# ---------------------------------------------------------------------------
def run_patient(row, img_enc, meta, heads, model_id, args, device) -> dict:
    cohort, pid = row["cohort"], int(row["SID"])
    print(f"\n[{cohort}/{row['case']}] SID {pid}  y={row['y']}  p={row['p']:.3f}")

    # `fold` is -1 whenever the whole-cohort head produced this patient's probability.
    head = heads["fold_heads"][row["fold"]] if row["fold"] >= 0 else heads["full_head"]
    z_cached = load_embeddings(model_id, cohort).loc[pid].values
    embed_dim = len(z_cached)
    beta_eff, member_share = patient_beta(head, z_cached, embed_dim)
    beta = torch.tensor(beta_eff, dtype=torch.float32, device=device)

    dataset = build_slice_dataset(cohort, pid, meta)
    if pid not in dataset._vols:
        raise FileNotFoundError(f"{cohort}/{pid}: no MRI found under the deployed path")
    n_slices = len(dataset)
    print(f"  {n_slices} slices, volume {dataset._vols[pid].shape}")

    c_s, saliency, z_bar, slice_ids, degenerate = slice_contributions_and_gradxinput(
        img_enc, dataset, pid, beta, device, args.batch_size
    )
    abs_mass = np.abs(c_s).sum()
    degen_mass = float(np.abs(c_s[degenerate]).sum() / abs_mass) if abs_mass > 0 else 0.0
    print(f"  degenerate (constant-input) slices: {degenerate.sum()}/{n_slices} "
          f"({degenerate.mean():.1%}), carrying {degen_mass:.1%} of Σ|c_s|")

    # Gate: the recomputed embedding must match the cache the thesis numbers came from.
    emb_gap = float(np.abs(z_bar - z_cached).max())
    if emb_gap > args.embedding_tol:
        raise RuntimeError(
            f"{cohort}/{pid}: recomputed z̄ differs from the cached embedding by "
            f"{emb_gap:.2e} (> {args.embedding_tol:.0e}). The attribution would not "
            f"describe the deployed model."
        )
    decomp_gap = float(abs(c_s.sum() - float(beta_eff @ z_cached)))
    print(f"  embedding gate {emb_gap:.2e} | Σc_s vs β·z̄ {decomp_gap:.2e}")

    # Top-|c_s| among slices that actually carry anatomy, with the most positive and most
    # negative forced in so the figures can always show both directions of the decision.
    # Degenerate slices are excluded here: they are constant images, so a per-voxel map of
    # them is meaningless — their influence is reported as `degenerate_c_s_fraction`.
    elig = np.flatnonzero(~degenerate)
    if elig.size == 0:
        raise RuntimeError(f"{cohort}/{pid}: every slice reaches the encoder as a "
                           f"constant image; nothing anatomical to attribute")
    ranked = elig[np.argsort(-np.abs(c_s[elig]))][: args.top_k_slices]
    order = list(ranked)
    for extreme in (int(elig[np.argmax(c_s[elig])]), int(elig[np.argmin(c_s[elig])])):
        if extreme not in order:
            order.append(extreme)
    order = np.array(order)

    ig_native, ig_delta, ig_resid = integrated_gradients_slices(
        img_enc, dataset, pid, beta, order.tolist(), meta, device,
        args.ig_steps, args.baseline, args.ig_batch_size,
    )
    # Per-slice completeness, each residual against its own target change.
    rel_per_slice = np.abs(ig_resid) / np.maximum(np.abs(ig_delta), 1e-12)
    rel_resid = float(rel_per_slice.max())
    rel_median = float(np.median(rel_per_slice))
    print(f"  IG top-{len(order)} completeness max residual {np.abs(ig_resid).max():.2e} "
          f"(relative: median {rel_median:.3f}, max {rel_resid:.3f})")

    tumour_per_slice = tumour_profile(cohort, pid, dataset, slice_ids)

    # --- persist ---
    tag = f"{cohort}_{pid}"
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    prof = pd.DataFrame({"slice": slice_ids, "c_s": c_s, "degenerate": degenerate})
    if tumour_per_slice is not None:
        prof["tumour_voxels"] = tumour_per_slice
    prof["is_top_k"] = prof.index.isin(order)
    prof.to_csv(out / f"slice_contributions_{tag}.csv", index=False)

    nib.save(
        nib.Nifti1Image(saliency, volume_affine(cohort, pid)),
        out / f"saliency_gradxinput_{tag}.nii.gz",
    )
    np.savez_compressed(
        out / f"ig_top_slices_{tag}.npz",
        slice_rows=order, slice_ids=slice_ids[order], ig=ig_native,
        c_s=c_s[order], delta=ig_delta, residual=ig_resid,
    )

    return {
        **{k: row[k] for k in ("cohort", "case", "case_desc", "case_order",
                               "SID", "y", "p", "fold")},
        "n_slices": n_slices,
        "sum_c_s": float(c_s.sum()),
        "beta_dot_z": float(beta_eff @ z_cached),
        "beta_eff_norm": float(np.linalg.norm(beta_eff)),
        "beta_eff_nonzero": int((beta_eff != 0).sum()),
        "member_grad_shares": [float(s) for s in member_share],
        "embedding_gate_max_abs": emb_gap,
        "decomposition_gap": decomp_gap,
        "n_degenerate": int(degenerate.sum()),
        "degenerate_fraction": float(degenerate.mean()),
        "degenerate_c_s_fraction": degen_mass,
        "top_slice": int(slice_ids[order[0]]),
        "top_slice_c_s": float(c_s[order[0]]),
        "max_pos_slice": int(slice_ids[int(elig[np.argmax(c_s[elig])])]),
        "max_neg_slice": int(slice_ids[int(elig[np.argmin(c_s[elig])])]),
        "ig_max_abs_residual": float(np.abs(ig_resid).max()),
        "ig_relative_residual": rel_resid,
        "ig_relative_residual_median": rel_median,
        "tumour_slices": int((tumour_per_slice > 0).sum()) if tumour_per_slice is not None else -1,
    }


# ---------------------------------------------------------------------------
# Screening: which confident hits peak near the liver?
# ---------------------------------------------------------------------------
def screen_candidates(p: pd.Series, y: pd.Series, kinds: str, max_n: int) -> pd.DataFrame:
    """Confident hits — ``y=1, p>0.5`` and ``y=0, p<0.5`` — ranked by confidence."""
    frames = []
    if kinds in ("tp", "all"):
        tp = pd.DataFrame({"p": p, "y": y}).query("y == 1 and p > 0.5")
        frames.append(tp.sort_values("p", ascending=False).assign(kind="tp"))
    if kinds in ("tn", "all"):
        tn = pd.DataFrame({"p": p, "y": y}).query("y == 0 and p < 0.5")
        frames.append(tn.sort_values("p", ascending=True).assign(kind="tn"))
    out = [f.head(max_n) if max_n > 0 else f for f in frames]
    return pd.concat(out).rename_axis("SID").reset_index()


def screen_cohort(cohort: str, img_enc, meta: dict, heads: dict, model_id: str,
                  args, device) -> pd.DataFrame:
    """One row per confident hit: where its extreme slices sit relative to the liver.

    The extremes are selected exactly as :func:`run_patient` selects them — the most
    positive and most negative ``c_s`` among non-degenerate slices — so a screened row
    predicts what the figures for that patient would show, without paying for Integrated
    Gradients. ``liver_score`` is the distance in mm from the extreme slices to the nearest
    tumour-bearing slice (``--liver-rank both`` takes the worse of the two, so a low score
    means *both* panels are at the liver); ``inf`` when either extreme carries no anatomy
    or when the patient has no usable tumour mask.
    """
    p, y = heads["scores"][cohort]
    cands = screen_candidates(p, y, args.screen_kinds, args.screen_max_n)
    print(f"\nScreening {len(cands)} confident hits in {cohort} "
          f"({(cands.kind == 'tp').sum()} tp, {(cands.kind == 'tn').sum()} tn)")

    rows = []
    for n, cand in enumerate(cands.itertuples(), 1):
        pid = int(cand.SID)
        z_cached = load_embeddings(model_id, cohort).loc[pid].values
        beta_eff, _ = patient_beta(heads["full_head"], z_cached, len(z_cached))
        beta = torch.tensor(beta_eff, dtype=torch.float32, device=device)

        dataset = build_slice_dataset(cohort, pid, meta)
        if pid not in dataset._vols:
            print(f"  [{n}/{len(cands)}] {pid}: no MRI under the deployed path, skipped")
            continue
        c_s, z_bar, slice_ids, degenerate = slice_contributions_only(
            img_enc, dataset, pid, beta, device, args.batch_size
        )
        emb_gap = float(np.abs(z_bar - z_cached).max())
        if emb_gap > args.embedding_tol:
            raise RuntimeError(f"{cohort}/{pid}: recomputed z̄ differs from the cached "
                               f"embedding by {emb_gap:.2e} (> {args.embedding_tol:.0e})")

        elig = np.flatnonzero(~degenerate)
        if elig.size == 0:
            print(f"  [{n}/{len(cands)}] {pid}: every slice is a constant image, skipped")
            continue
        pos = int(elig[np.argmax(c_s[elig])])
        neg = int(elig[np.argmin(c_s[elig])])

        vol = dataset._vols[pid]
        fg = slice_foreground_fraction(vol, dataset.axis, slice_ids,
                                       args.screen_fg_rel_thresh)
        tum = tumour_profile(cohort, pid, dataset, slice_ids)
        dist = tumour_slice_distance(tum)
        mm = slice_spacing_mm(cohort, pid, dataset.axis)

        d_pos = float(dist[pos]) * mm if dist is not None else np.nan
        d_neg = float(dist[neg]) * mm if dist is not None else np.nan
        legible = fg[pos] >= args.screen_fg_floor and fg[neg] >= args.screen_fg_floor
        if args.liver_rank == "pos":
            legible = fg[pos] >= args.screen_fg_floor
            score = d_pos
        else:
            score = max(d_pos, d_neg)
        score = float(score) if (legible and np.isfinite(score)) else float("inf")

        abs_mass = np.abs(c_s).sum()
        hit = np.flatnonzero(tum > 0) if tum is not None else np.array([], dtype=int)
        rows.append({
            "cohort": cohort, "SID": pid, "kind": cand.kind,
            "y": int(cand.y), "p": float(cand.p),
            "n_slices": len(c_s),
            "n_degenerate": int(degenerate.sum()),
            "degenerate_c_s_fraction":
                float(np.abs(c_s[degenerate]).sum() / abs_mass) if abs_mass > 0 else 0.0,
            "slice_spacing_mm": mm,
            "max_pos_slice": int(slice_ids[pos]), "max_pos_c_s": float(c_s[pos]),
            "max_pos_fg_frac": float(fg[pos]),
            "max_pos_dist_slices": float(dist[pos]) if dist is not None else np.nan,
            "max_pos_dist_mm": d_pos,
            "max_neg_slice": int(slice_ids[neg]), "max_neg_c_s": float(c_s[neg]),
            "max_neg_fg_frac": float(fg[neg]),
            "max_neg_dist_slices": float(dist[neg]) if dist is not None else np.nan,
            "max_neg_dist_mm": d_neg,
            "tumour_first_slice": int(slice_ids[hit[0]]) if hit.size else -1,
            "tumour_last_slice": int(slice_ids[hit[-1]]) if hit.size else -1,
            "liver_score": score,
        })
        print(f"  [{n}/{len(cands)}] {pid} ({cand.kind}, p={cand.p:.3f}): "
              f"pos {slice_ids[pos]} (fg {fg[pos]:.1%}, {d_pos:.0f} mm), "
              f"neg {slice_ids[neg]} (fg {fg[neg]:.1%}, {d_neg:.0f} mm) "
              f"-> liver_score {score:.0f}")

    return pd.DataFrame(rows).sort_values(["liver_score", "kind"]).reset_index(drop=True)


def run_screen(args) -> pd.DataFrame:
    """``--screen``: rank confident hits by how close their extremes land to the liver."""
    device = torch.device(args.device)
    specs = load_members(args.members_csv)
    _cases, heads = build_case_table(
        args.model_id, specs, args.target, args.cohorts,
        resection_head=args.resection_head,
    )
    img_enc, meta = load_contrastive_model(args.model_id, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    out = []
    for cohort in args.cohorts:
        screen = screen_cohort(cohort, img_enc, meta, heads, args.model_id, args, device)
        path = args.output_dir / f"saliency_screen_{cohort}.csv"
        screen.to_csv(path, index=False)
        print(f"\nWrote {path}")
        best = screen[np.isfinite(screen["liver_score"])]
        for kind in ("tp", "tn"):
            k = best[best["kind"] == kind]
            if len(k):
                b = k.iloc[0]
                print(f"  best {kind}: SID {int(b.SID)}  p={b.p:.3f}  "
                      f"liver_score {b.liver_score:.0f} mm  "
                      f"(pos {int(b.max_pos_slice)}, neg {int(b.max_neg_slice)})")
        out.append(screen)
    return pd.concat(out, ignore_index=True)


def parse_pins(specs: list[str]) -> dict[str, dict[str, int]]:
    """``cohort:label=SID`` strings -> ``{cohort: {label: SID}}``."""
    pinned: dict[str, dict[str, int]] = {}
    for spec in specs or []:
        try:
            cohort_label, sid = spec.split("=")
            cohort, label = cohort_label.split(":")
        except ValueError:
            raise SystemExit(f"--pin-case expects cohort:label=SID, got {spec!r}")
        pinned.setdefault(cohort, {})[label] = int(sid)
    return pinned


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _merge_existing(out_dir: Path, summary: pd.DataFrame, cases: pd.DataFrame,
                    cohorts: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Fold this run's cohorts into the index files already in ``out_dir``.

    Rows for a cohort computed in this run replace any older rows for the same cohort,
    so re-running one cohort with different settings refreshes it rather than duplicating
    it. Cohorts keep the order they were first written in, with the new ones appended.
    """
    prev_meta = out_dir / "saliency_meta.json"
    prev_summary = out_dir / "saliency_summary.csv"
    if not (prev_summary.exists() and (out_dir / "saliency_cases.csv").exists()):
        return summary, cases, cohorts

    fresh = set(summary["cohort"])
    old = pd.read_csv(prev_summary)
    summary = pd.concat([old[~old["cohort"].isin(fresh)], summary], ignore_index=True)
    old_cases = pd.read_csv(out_dir / "saliency_cases.csv")
    cases = pd.concat([old_cases[~old_cases["cohort"].isin(fresh)], cases],
                      ignore_index=True)

    prev = json.loads(prev_meta.read_text()).get("cohorts", []) if prev_meta.exists() else []
    merged = prev + [c for c in cohorts if c not in prev]
    print(f"  appended to {out_dir}: cohorts {merged}")
    return summary, cases, merged


def run(args) -> pd.DataFrame:
    device = torch.device(args.device)
    specs = load_members(args.members_csv)
    print(f"Head members: " + ", ".join(
        f"{s['model']}/{s['fs']} k={s['select_k']}" for s in specs))

    pinned = parse_pins(args.pin_case)
    cases, heads = build_case_table(
        args.model_id, specs, args.target, args.cohorts,
        resection_head=args.resection_head,
        extra_tp=args.extra_tp, extra_tn=args.extra_tn, pinned=pinned,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.append:   # in append mode the previous file is needed for the merge
        cases.to_csv(args.output_dir / "saliency_cases.csv", index=False)
    print(f"\n{len(cases)} cases selected:")
    print(cases[["cohort", "case", "SID", "y", "p"]].to_string(index=False))

    # Head reconstruction check, mirroring the gene-side report.
    X_res = heads["X_res"]
    params_full = unwind_ensemble(heads["full_head"], X_res.shape[1])
    head_recon = float(np.max(np.abs(
        ensemble_score(params_full, X_res.values)
        - heads["full_head"].predict_proba(X_res)[:, 1]
    )))
    print(f"\nHead reconstruction max residual: {head_recon:.2e}")

    img_enc, meta = load_contrastive_model(args.model_id, device)
    print(f"Encoder {args.model_id}: {meta['model']}, axes={meta['axes']}, "
          f"n_per_axis={meta['n_per_axis']}, mri_type={meta['mri_type']}")

    rows = [run_patient(r, img_enc, meta, heads, args.model_id, args, device)
            for _, r in cases.iterrows()]
    summary = pd.DataFrame(rows)
    cohorts = list(args.cohorts)
    if args.append:
        # Adding a cohort to an existing run: the per-patient artefacts are already on
        # disk for the cohorts done earlier, and every stage is deterministic, so only
        # the index files have to be reconciled.
        summary, cases, cohorts = _merge_existing(args.output_dir, summary, cases, cohorts)
    summary.to_csv(args.output_dir / "saliency_summary.csv", index=False)
    cases.to_csv(args.output_dir / "saliency_cases.csv", index=False)

    (args.output_dir / "saliency_meta.json").write_text(json.dumps({
        "model_id": args.model_id,
        "members_csv": str(args.members_csv),
        "members": specs,
        "target": args.target,
        "cohorts": cohorts,
        "resection_head": args.resection_head,
        "extra_tp": args.extra_tp,
        "extra_tn": args.extra_tn,
        "pin_case": list(args.pin_case or []),
        "top_k_slices": args.top_k_slices,
        "ig_steps": args.ig_steps,
        "baseline": args.baseline,
        "head_reconstruction_max_residual": head_recon,
        "encoder": {k: meta[k] for k in ("model", "axes", "n_per_axis", "img_size",
                                         "mri_type", "freeze_backbone", "lam")},
    }, indent=2))
    print(f"\nWrote {args.output_dir}/saliency_summary.csv")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-id", default="d7085bf5")
    p.add_argument("--members-csv", type=Path,
                   default=Path("results/eval/grid_flat3_bestckpt/d7085bf5/"
                                "model_ensemble_members.csv"))
    p.add_argument("--cohorts", nargs="+", default=["resection", "soramic"])
    p.add_argument("--append", action="store_true",
                   help="merge these cohorts into the summary/case index already in "
                        "--output-dir instead of replacing it, so a cohort can be added "
                        "to an existing run without recomputing the others. Rows for a "
                        "cohort present in this run replace their older counterparts")
    p.add_argument("--target", default="rfs_2year")
    p.add_argument("--resection-head", choices=["full", "oof"], default="full",
                   help="which head scores and attributes the resection cohort. "
                        "full: the head fit on the whole cohort, so every panel shares "
                        "one head and matches Soramic — but p is in-sample and "
                        "optimistic. oof: out-of-fold p, each patient attributed with "
                        "its own held-out fold's head")
    p.add_argument("--extra-tp", type=int, default=2,
                   help="additional next-ranked true positives (tp_high_2, ...)")
    p.add_argument("--extra-tn", type=int, default=2,
                   help="additional next-ranked true negatives (tn_low_2, ...)")
    p.add_argument("--pin-case", action="append", metavar="COHORT:LABEL=SID",
                   help="add a case chosen outside the probability ranking, e.g. "
                        "`resection:tp_liver=53` for a patient the --screen pass picked. "
                        "Repeatable. Pinned labels are kept distinct so the rank-based "
                        "findings are not computed over hand-picked cases")
    p.add_argument("--screen", action="store_true",
                   help="rank every confident hit by how close its extreme slices land to "
                        "the liver and exit, writing saliency_screen_<cohort>.csv. "
                        "Forward-only: no IG, no saliency volume, no index files touched")
    p.add_argument("--screen-kinds", choices=["tp", "tn", "all"], default="all",
                   help="which confident hits to screen")
    p.add_argument("--screen-max-n", type=int, default=0,
                   help="cap on candidates per kind, most confident first (0 = all)")
    p.add_argument("--screen-fg-rel-thresh", type=float, default=0.10,
                   help="a voxel counts as anatomy above this fraction of the volume's p99")
    p.add_argument("--screen-fg-floor", type=float, default=0.02,
                   help="an extreme slice below this foreground fraction is body wall or "
                        "air; the patient is scored `inf` rather than ranked")
    p.add_argument("--liver-rank", choices=["both", "pos"], default="both",
                   help="both: score by the worse of the two extremes, so a good score "
                        "means both panels are at the liver. pos: score by the most "
                        "positive slice only")
    p.add_argument("--device", default="cpu",
                   help="cpu by default: MPS is ~60x slower than CPU for this backward pass")
    p.add_argument("--top-k-slices", type=int, default=16)
    p.add_argument("--ig-steps", type=int, default=256,
                   help="IG converges slowly on this frozen ViT; 256 gives ~1-3%% "
                        "completeness residual with the blur baseline")
    p.add_argument("--baseline", choices=["zero", "blur"], default="blur",
                   help="blur: a uniform baseline sits in LayerNorm's near-singular "
                        "region, so 'zero' does not reach completeness at any practical "
                        "step count (0.34-0.73 residual at 256 steps)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--ig-batch-size", type=int, default=16)
    p.add_argument("--embedding-tol", type=float, default=1e-4)
    p.add_argument("--output-dir", type=Path,
                   default=Path("results/eval/interpretability/image_saliency/d7085bf5"))
    return p.parse_args()


if __name__ == "__main__":
    _args = parse_args()
    run_screen(_args) if _args.screen else run(_args)
