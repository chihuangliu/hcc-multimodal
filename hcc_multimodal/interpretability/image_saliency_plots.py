"""Figures for :mod:`hcc_multimodal.interpretability.image_saliency`.

All read back from the runner's ``--output-dir``. This module draws **figures only** — the
report itself is written by hand. The two voxel-level figures are drawn for the
``--main-cases`` set; everything else is appendix material:

1. ``top_slices_<cohort>`` — the most positive and the most negative contributing slice
   per patient, with the Integrated-Gradients map overlaid and the tumour contoured.
2. ``saliency_mip_<cohort>`` — maximum-intensity projection of the Gradient×Input volume
   along the slicing axis over the anatomical MIP.
3. ``slice_profile_<cohort>`` (appendix) — per-slice contribution ``c_s`` against slice
   index, with the tumour's extent along the slicing axis shaded and constant-input
   slices greyed. Drawn for every case, main and extra.
4. ``top_slices_<cohort>_extra`` / ``saliency_mip_<cohort>_extra`` (appendix) — the same
   two figures for the additional rank-2/3 confident hits, kept out of the main text so
   each category is represented there by a single exemplar.
5. ``top_slices_<cohort>_liver`` / ``saliency_mip_<cohort>_liver`` — the same two figures
   for the ``--liver-cases`` pins: confident hits the runner's ``--screen`` pass found to
   have their extreme slices nearest the liver. They are selected for anatomical
   legibility rather than by probability rank, so they are kept in their own figures and
   must stay out of any count the report makes over the rank-selected cases.

Slices are drawn transposed (slicing axis out of the page, third axis vertical) with the
physical 1x1x3 mm aspect applied, so the panels are anatomically proportioned. Saved as
editable SVG plus PNG, matching ``hcc_multimodal/survival/plots.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"

import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from hcc_multimodal.eval.data import _normalize_slice  # noqa: E402
from hcc_multimodal.interpretability.image_saliency import (  # noqa: E402
    CASES,
    build_slice_dataset,
    load_mask,
)

_POS, _NEG = "#c44e52", "#4c72b0"
_ASPECT = 3.0        # 3 mm slice thickness against 1 mm in-plane
_PATCH_GRID = 7      # ViT-B/32: 224px input / 32px patches
_OVERLAY_FLOOR = 0.10  # hide patches below this fraction of the colour scale


def _save(fig, fig_dir: Path, stem: str) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(fig_dir / f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {fig_dir / stem}.png/.svg")


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (start, end) index pairs of each contiguous True run."""
    out, start = [], None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        elif not f and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(flags) - 1))
    return out


def _panel_title(row: pd.Series) -> str:
    return (f"{row['case']} — SID {row['SID']}\n"
            f"y={row['y']}, p={row['p']:.3f}")


def _show_slice(ax, img: np.ndarray) -> None:
    """Greyscale anatomical slice, transposed and physically proportioned.

    The 3:1 aspect makes matplotlib shrink the axes box inside its grid cell, and by
    different amounts per panel because the slices differ in shape. Anchoring north
    keeps every panel's title at the top of its own cell instead of floating down into
    the panel above it.
    """
    ax.imshow(img.T, cmap="gray", aspect=_ASPECT, origin="lower")
    ax.set_anchor("N")
    ax.set_xticks([])
    ax.set_yticks([])


def _patch_pool(attr: np.ndarray, grid: int = _PATCH_GRID,
                crop_frac: float = 224.0 / 256.0) -> np.ndarray:
    """Sum the attribution within each ViT patch footprint, held block-constant.

    ViT-B/32 sees a 7x7 grid of patches over its 224px input, so a per-pixel map claims
    a spatial precision the encoder does not have. Pooling to that grid shows what the
    model can actually localise. The centre crop is undone first: the model's 224 input
    is the central ``crop_frac`` of the slice, and the border outside it stays zero.
    """
    H, W = attr.shape
    h, w = int(round(H * crop_frac)), int(round(W * crop_frac))
    y0, x0 = (H - h) // 2, (W - w) // 2
    box = attr[y0:y0 + h, x0:x0 + w]

    ys = np.linspace(0, h, grid + 1).astype(int)
    xs = np.linspace(0, w, grid + 1).astype(int)
    out = np.zeros_like(attr)
    for i in range(grid):
        for j in range(grid):
            out[y0 + ys[i]:y0 + ys[i + 1], x0 + xs[j]:x0 + xs[j + 1]] = (
                box[ys[i]:ys[i + 1], xs[j]:xs[j + 1]].sum()
            )
    return out


def _overlay(ax, attr: np.ndarray, pct: float = 99.0, pool: bool = True,
             floor: float = _OVERLAY_FLOOR) -> float:
    """Diverging saliency overlay, symmetric about zero at the ``pct`` percentile.

    ``floor`` hides patches below that fraction of the colour scale. It is a *relative*
    cut, so how much it hides depends on how peaked the attribution is: a map with one
    dominant region keeps only that region, while a flat one stays almost fully painted.
    """
    if pool:
        attr = _patch_pool(attr)
    v = np.percentile(np.abs(attr), pct)
    v = float(v) if v > 0 else float(np.abs(attr).max() or 1.0)
    masked = np.ma.masked_where(np.abs(attr) < floor * v, attr)
    ax.imshow(masked.T, cmap="bwr", vmin=-v, vmax=v, alpha=0.50,
              aspect=_ASPECT, origin="lower")
    return v


def _contour(ax, mask_slice: np.ndarray) -> None:
    if mask_slice is not None and mask_slice.any():
        ax.contour(mask_slice.T, levels=[0.5], colors="#2ca02c", linewidths=1.0)


# ---------------------------------------------------------------------------
# Figure 1 — per-slice contribution profiles
# ---------------------------------------------------------------------------
def fig_slice_profile(cohort: str, rows: pd.DataFrame, in_dir: Path, fig_dir: Path) -> None:
    n = len(rows)
    fig, axes = plt.subplots(n, 1, figsize=(9, 2.0 * n), sharex=False)
    axes = np.atleast_1d(axes)

    for ax, (_, row) in zip(axes, rows.iterrows()):
        prof = pd.read_csv(in_dir / f"slice_contributions_{cohort}_{row['SID']}.csv")
        c = prof["c_s"].values
        ax.axhline(0, color="0.6", lw=0.8)
        ax.fill_between(prof["slice"], 0, c, where=c >= 0, color=_POS, alpha=0.75, lw=0)
        ax.fill_between(prof["slice"], 0, c, where=c < 0, color=_NEG, alpha=0.75, lw=0)

        if "tumour_voxels" in prof and prof["tumour_voxels"].sum() > 0:
            tum = prof.loc[prof["tumour_voxels"] > 0, "slice"]
            ax.axvspan(tum.min(), tum.max(), color="#2ca02c", alpha=0.15, lw=0,
                       label="tumour extent")

        # Constant-input slices: no anatomy, but still inside the mean-pooled embedding.
        # Drawn as contiguous runs — these come in blocks at the volume edges.
        if "degenerate" in prof and prof["degenerate"].any():
            first = True
            for lo, hi in _runs(prof["degenerate"].values):
                ax.axvspan(prof["slice"].values[lo] - 0.5, prof["slice"].values[hi] + 0.5,
                           color="0.35", alpha=0.30, lw=0,
                           label="degenerate (constant) slice" if first else None)
                first = False
        ax.legend(loc="upper right", fontsize=7, framealpha=0.85)

        ax.set_ylabel("$c_s$", fontsize=9)
        ax.set_title(_panel_title(row).replace("\n", "  |  "), fontsize=9, loc="left")
        ax.tick_params(labelsize=8)
    axes[-1].set_xlabel("sagittal slice index", fontsize=9)

    fig.suptitle(
        f"Per-slice contribution to the risk score — {cohort}\n"
        r"$c_s=\beta_{\mathrm{eff}}\cdot f(x_s)/S$,  $\sum_s c_s=\beta_{\mathrm{eff}}\cdot\bar z$",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, fig_dir, f"slice_profile_{cohort}")


# ---------------------------------------------------------------------------
# Figure 2 — IG on the extreme slices
# ---------------------------------------------------------------------------
def fig_top_slices(cohort: str, rows: pd.DataFrame, in_dir: Path, fig_dir: Path,
                   meta: dict, mask_overlay: bool, stem: str | None = None,
                   floor: float = _OVERLAY_FLOOR) -> None:
    """One row per patient: the most positive slice and the most negative slice.

    Like :func:`fig_saliency_mip`, the 2D content is gathered first so the grid rows can
    be sized to their own slice shape rather than all being forced equal.
    """
    panels = []
    for _, row in rows.iterrows():
        pid = int(row["SID"])
        npz = np.load(in_dir / f"ig_top_slices_{cohort}_{pid}.npz")
        ds = build_slice_dataset(cohort, pid, meta)
        vol, axis = ds._vols[pid], ds.axis
        mask = load_mask(cohort, pid) if mask_overlay else None
        if mask is not None and mask.shape != vol.shape:
            mask = None

        c_top = npz["c_s"]
        picks = [(int(np.argmax(c_top)), "most positive"),
                 (int(np.argmin(c_top)), "most negative")]
        cells = []
        for idx, label in picks:
            si = int(npz["slice_ids"][idx])
            cells.append({
                "img": _normalize_slice(np.take(vol, si, axis=axis)),
                "ig": npz["ig"][idx],
                "mask": np.take(mask, si, axis=axis) if mask is not None else None,
                "title": f"{label} — slice {si}\n$c_s$={c_top[idx]:+.4f}",
            })
        panels.append({"cells": cells, "label": _panel_title(row)})

    n, ncol = len(panels), 2
    ratios = [_ASPECT * p["cells"][0]["img"].shape[1] / p["cells"][0]["img"].shape[0]
              for p in panels]
    panel_w = 4.5
    fig, axes = plt.subplots(
        n, ncol, figsize=(panel_w * ncol, panel_w * sum(ratios) + 0.85 * n + 0.8),
        gridspec_kw={"height_ratios": ratios}, layout="constrained",
    )
    axes = np.atleast_2d(axes)

    for r, p in enumerate(panels):
        for col, cell in enumerate(p["cells"]):
            ax = axes[r, col]
            _show_slice(ax, cell["img"])
            _overlay(ax, cell["ig"], floor=floor)
            _contour(ax, cell["mask"])
            ax.set_title(cell["title"], fontsize=8)
        axes[r, 0].set_ylabel(p["label"], fontsize=8)

    fig.suptitle(
        f"Integrated-Gradients attribution on the extreme slices — {cohort}\n"
        "pooled to the encoder's 7×7 patch grid; red = pushes toward recurrence ≤ 2 yr, "
        "blue = away; green contour = tumour",
        fontsize=11,
    )
    _save(fig, fig_dir, stem or f"top_slices_{cohort}")


# ---------------------------------------------------------------------------
# Figure 3 — saliency MIP
# ---------------------------------------------------------------------------
def fig_saliency_mip(cohort: str, rows: pd.DataFrame, in_dir: Path, fig_dir: Path,
                     meta: dict, mask_overlay: bool, ncol: int = 3,
                     stem: str | None = None, floor: float = _OVERLAY_FLOOR) -> None:
    """MIP of the Gradient×Input volume along the slicing axis, over the anatomical MIP.

    The 2D projections are collected before the figure is created so each grid row can be
    given a height proportional to its tallest panel. Without that, panels of different
    slice shape shrink by different amounts inside equal-height cells and the titles of
    one row end up drawn over the images of the row above.
    """
    panels = []
    for _, row in rows.iterrows():
        pid = int(row["SID"])
        sal = np.asarray(nib.load(
            in_dir / f"saliency_gradxinput_{cohort}_{pid}.nii.gz").dataobj)
        ds = build_slice_dataset(cohort, pid, meta)
        vol, axis = ds._vols[pid], ds.axis

        pos_mip = np.maximum(sal, 0).max(axis=axis)
        neg_mip = np.minimum(sal, 0).min(axis=axis)
        mask = load_mask(cohort, pid) if mask_overlay else None
        panels.append({
            "anat": _normalize_slice(vol.max(axis=axis)),
            # Each ray keeps whichever extreme is larger in magnitude, so a positive and
            # a negative peak on the same ray do not average into nothing.
            "signed": np.where(pos_mip >= -neg_mip, pos_mip, neg_mip),
            "mask": mask.max(axis=axis)
                    if (mask is not None and mask.shape == vol.shape) else None,
            "title": _panel_title(row),
        })

    n = len(panels)
    nrow = int(np.ceil(n / ncol))
    # Displayed height/width of a panel: the array is drawn transposed at 3:1 aspect.
    ratios = [_ASPECT * p["anat"].shape[1] / p["anat"].shape[0] for p in panels]
    row_h = [max(ratios[r * ncol:(r + 1) * ncol]) for r in range(nrow)]
    panel_w = 4.2
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(panel_w * ncol, panel_w * sum(row_h) + 0.95 * nrow + 0.6),
        gridspec_kw={"height_ratios": row_h}, layout="constrained",
    )
    axes = np.atleast_1d(axes).ravel()

    for ax, p in zip(axes, panels):
        _show_slice(ax, p["anat"])
        _overlay(ax, p["signed"], floor=floor)
        _contour(ax, p["mask"])
        # The tallest panel in a row fills its cell exactly, so the title needs its own
        # pad or the second line sits flush on the image.
        ax.set_title(p["title"], fontsize=9, pad=8)
    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(
        f"Gradient×Input saliency, maximum-intensity projection along the slicing axis — {cohort}",
        fontsize=11,
    )
    _save(fig, fig_dir, stem or f"saliency_mip_{cohort}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args) -> None:
    meta_all = json.loads((args.input_dir / "saliency_meta.json").read_text())
    meta = meta_all["encoder"]
    summary = pd.read_csv(args.input_dir / "saliency_summary.csv")
    # Cohorts appear in the order they were requested of the runner, not alphabetically,
    # so the report keeps resection (the training cohort) first.
    rank = {c: i for i, c in enumerate(meta_all.get("cohorts")
                                       or summary["cohort"].unique())}
    summary = (summary.assign(_ord=summary["cohort"].map(lambda c: rank.get(c, len(rank))))
               .sort_values(["_ord", "case_order"]).drop(columns="_ord"))

    # Three-way: the screened pins get their own section, so they must not fall through
    # into Appendix D with the rank-2/3 extras.
    is_liver = summary["case"].isin(args.liver_cases)
    liver = summary[is_liver]
    rest = summary[~is_liver]
    is_main = rest["case"].isin(args.main_cases)
    main, extra = rest[is_main], rest[~is_main]
    if not len(main):
        raise SystemExit(f"--main-cases {args.main_cases} matched no rows; "
                         f"available: {sorted(summary['case'].unique())}")

    for cohort in summary["cohort"].unique():
        rows = summary[summary["cohort"] == cohort]
        m = main[main["cohort"] == cohort]
        e = extra[extra["cohort"] == cohort]
        v = liver[liver["cohort"] == cohort]
        print(f"{cohort}: {len(m)} main + {len(e)} appendix + {len(v)} liver cases")
        fig_slice_profile(cohort, rows, args.input_dir, args.fig_dir)
        for subset, suffix in ((m, ""), (e, "_extra"), (v, "_liver")):
            if not len(subset):
                continue
            fig_top_slices(cohort, subset, args.input_dir, args.fig_dir, meta,
                           args.mask_overlay, stem=f"top_slices_{cohort}{suffix}",
                           floor=args.overlay_floor)
            fig_saliency_mip(cohort, subset, args.input_dir, args.fig_dir, meta,
                             args.mask_overlay, stem=f"saliency_mip_{cohort}{suffix}",
                             floor=args.overlay_floor)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", type=Path,
                   default=Path("results/eval/interpretability/image_saliency/d7085bf5"))
    p.add_argument("--fig-dir", type=Path, default=Path("reports/0810/image_saliency"))
    p.add_argument("--no-mask-overlay", dest="mask_overlay", action="store_false")
    p.add_argument("--overlay-floor", type=float, default=_OVERLAY_FLOOR,
                   help="hide saliency patches below this fraction of the colour "
                        "scale. Relative, so its effect depends on how peaked the "
                        "attribution is: raise it when a flat map paints every patch")
    p.add_argument("--main-cases", nargs="+", default=list(CASES),
                   help="cases carried by the main text's two voxel-level figures, one "
                        "exemplar per outcome x prediction category. Everything else "
                        "(the rank-2/3 confident hits) goes to Appendix D")
    p.add_argument("--liver-cases", nargs="+", default=["tp_liver", "tn_liver"],
                   help="cases pinned by the runner's --screen pass for having their "
                        "extreme slices at the liver. They get their own `_liver` figures "
                        "rather than falling into the appendix set, because they were "
                        "chosen for anatomical legibility and not by probability rank")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
