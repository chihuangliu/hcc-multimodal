"""Figures for :mod:`hcc_multimodal.interpretability.image_saliency`.

All read back from the runner's ``--output-dir``. The two voxel-level figures carry the
report's main text and are drawn for the ``--main-cases`` set only; everything else is
appendix material:

1. ``top_slices_<cohort>`` — the most positive and most negative contributing slice per
   patient, with the Integrated-Gradients map overlaid and the tumour contoured.
2. ``saliency_mip_<cohort>`` — maximum-intensity projection of the Gradient×Input volume
   along the slicing axis over the anatomical MIP.
3. ``slice_profile_<cohort>`` (appendix) — per-slice contribution ``c_s`` against slice
   index, with the tumour's extent along the slicing axis shaded and constant-input
   slices greyed. Drawn for every case, main and extra.
4. ``top_slices_<cohort>_extra`` / ``saliency_mip_<cohort>_extra`` (appendix) — the same
   two figures for the additional rank-2/3 confident hits, kept out of the main text so
   each category is represented there by a single exemplar.

Slices are drawn transposed (slicing axis out of the page, third axis vertical) with the
physical 1x1x3 mm aspect applied, so the panels are anatomically proportioned. Saved as
editable SVG plus PNG, matching ``hcc_multimodal/survival/plots.py``.
"""

from __future__ import annotations

import argparse
import json
import os
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


def _overlay(ax, attr: np.ndarray, pct: float = 99.0, pool: bool = True) -> float:
    """Diverging saliency overlay, symmetric about zero at the ``pct`` percentile."""
    if pool:
        attr = _patch_pool(attr)
    v = np.percentile(np.abs(attr), pct)
    v = float(v) if v > 0 else float(np.abs(attr).max() or 1.0)
    masked = np.ma.masked_where(np.abs(attr) < 0.10 * v, attr)
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
                   meta: dict, mask_overlay: bool, stem: str | None = None) -> None:
    """One row per patient: reference anatomy, most positive slice, most negative slice.

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
        # Column 1 is the strongest slice either way, shown clean as an anatomical
        # reference; it usually coincides with column 2 or 3.
        picks = [(int(np.argmax(np.abs(c_top))), "anatomy", False),
                 (int(np.argmax(c_top)), "most positive", True),
                 (int(np.argmin(c_top)), "most negative", True)]
        cells = []
        for idx, label, show_ig in picks:
            si = int(npz["slice_ids"][idx])
            cells.append({
                "img": _normalize_slice(np.take(vol, si, axis=axis)),
                "ig": npz["ig"][idx] if show_ig else None,
                "mask": np.take(mask, si, axis=axis) if mask is not None else None,
                "title": f"{label} — slice {si}\n$c_s$={c_top[idx]:+.4f}",
            })
        panels.append({"cells": cells, "label": _panel_title(row)})

    n = len(panels)
    ratios = [_ASPECT * p["cells"][0]["img"].shape[1] / p["cells"][0]["img"].shape[0]
              for p in panels]
    panel_w = 3.5
    fig, axes = plt.subplots(
        n, 3, figsize=(panel_w * 3, panel_w * sum(ratios) + 0.85 * n + 0.8),
        gridspec_kw={"height_ratios": ratios}, layout="constrained",
    )
    axes = np.atleast_2d(axes)

    for r, p in enumerate(panels):
        for col, cell in enumerate(p["cells"]):
            ax = axes[r, col]
            _show_slice(ax, cell["img"])
            if cell["ig"] is not None:
                _overlay(ax, cell["ig"])
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
                     stem: str | None = None) -> None:
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
        _overlay(ax, p["signed"])
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
def peak_vs_tumour(summary: pd.DataFrame, in_dir: Path) -> pd.DataFrame:
    """Where each patient's extreme slices sit relative to the tumour's slice extent.

    Read straight off the same profile data the §5 figures plot — no voxel-level masking.
    """
    rows = []
    for _, r in summary.iterrows():
        prof = pd.read_csv(in_dir / f"slice_contributions_{r['cohort']}_{r['SID']}.csv")
        if "tumour_voxels" not in prof or prof["tumour_voxels"].sum() == 0:
            rows.append({"lo": None, "hi": None, "pos_in": None, "neg_in": None})
            continue
        tum = prof.loc[prof["tumour_voxels"] > 0, "slice"]
        lo, hi = int(tum.min()), int(tum.max())
        rows.append({
            "lo": lo, "hi": hi,
            "pos_in": lo <= r["max_pos_slice"] <= hi,
            "neg_in": lo <= r["max_neg_slice"] <= hi,
        })
    return pd.DataFrame(rows, index=summary.index)


def _case_table(rows: pd.DataFrame) -> list[str]:
    """The per-case summary table body, shared by the main text and Appendix D."""
    L = ["| Cohort | Case | SID | y | p | slices | Σc_s | ‖β_eff‖ | nnz | "
         "max-pos slice | max-neg slice | tumour slices |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in rows.iterrows():
        L.append(
            f"| {r['cohort']} | {r['case']} | {r['SID']} | {r['y']} | {r['p']:.3f} | "
            f"{r['n_slices']} | {r['sum_c_s']:+.4f} | {r['beta_eff_norm']:.1f} | "
            f"{r['beta_eff_nonzero']} | {r['max_pos_slice']} | {r['max_neg_slice']} | "
            f"{r['tumour_slices']} |"
        )
    return L


def write_report(summary: pd.DataFrame, main: pd.DataFrame, extra: pd.DataFrame,
                 meta_all: dict, in_dir: Path, fig_dir: Path, report: Path) -> None:
    """Numeric skeleton of the report.

    The main text carries one exemplar per outcome × prediction category and the two
    voxel-level figures. The per-slice profile, the tumour-overlap table, the
    constant-input slice analysis and the additional rank-2/3 confident hits are
    appendices, so the main narrative stays on *what the attribution shows about a
    single representative patient per category*.
    """
    enc = meta_all["encoder"]
    members = ", ".join(f"`{m['model']}`/`{m['fs']}` k={m['select_k']}"
                        for m in meta_all["members"])
    figrel = os.path.relpath(fig_dir, report.parent)
    cohorts = list(summary["cohort"].unique())

    L = [f"# Image Saliency — {pd.Timestamp.today():%Y-%m-%d}", ""]
    L.append(
        f"Attributes the deployed 2-year RFS **model ensemble** ({members}) on run "
        f"`{meta_all['model_id']}` back to MRI voxels, through the frozen image encoder. "
        f"The main text shows one exemplar per outcome × prediction category "
        f"({len(main)} patients across {', '.join('`' + c + '`' for c in cohorts)})."
        + (f" A further {len(extra)} confident hits, taken at the next probability ranks, "
           f"are in Appendix D." if len(extra) else "")
    )
    if meta_all.get("resection_head", "oof") == "full":
        L += ["", "> **Head.** Both cohorts are scored and attributed by the single "
              "downstream head fit on the **whole** resection cohort. Resection `p` is "
              "therefore **in-sample and optimistic** — it is not a performance estimate "
              "and does not correspond to the cross-validated AUROC in the thesis tables. "
              "It is used here only to rank patients into the six categories, so that "
              "every panel in this report is attributing the same head. Pass "
              "`--resection-head oof` for out-of-fold probabilities with each resection "
              "patient attributed by its own held-out fold's head."]

    # --- findings, computed from the same data the figures plot ---
    pk = peak_vs_tumour(summary, in_dir)
    res = summary["cohort"] == "resection"
    sor = summary["cohort"] == "soramic"

    L += ["", "## 1. Key findings", ""]
    # 1. Do the model's strongest slices sit on the lesion, and does that separate the
    #    cases it gets right from the ones it gets wrong? Counted, not asserted.
    hit = summary["case"].str.startswith(("tp_", "tn_")).values
    scored = pk["pos_in"].notna().values

    def _tally(sel):
        s = sel & scored
        return int(pk.loc[s, "pos_in"].sum()), int(s.sum())

    h_in, h_n = _tally(hit)
    m_in, m_n = _tally(~hit)
    hs_in, hs_n = _tally(hit & sor.values)
    L.append(
        f"1. **The model's strongest evidence usually does not sit on the lesion, and "
        f"that does not separate hits from misses.** Pooling every patient run, including "
        f"the Appendix D extras, the most positive slice falls inside the tumour's slice "
        f"extent for **{h_in} of {h_n}** confident hits (`tp_*`/`tn_*`) and **{m_in} of "
        f"{m_n}** misses — on Soramic alone, {hs_in} of {hs_n} hits. With only a handful "
        f"of exemplars per cell this is an observation, not a test; the per-case detail "
        f"is in Appendix B."
    )
    # 2. Peripheral slices: how often is the extreme slice in the outermost 10% of the
    #    volume, where a sagittal slice is body wall, air or an off-anatomy artefact?
    edge = np.minimum(summary["max_neg_slice"], summary["n_slices"] - 1
                      - summary["max_neg_slice"]) / summary["n_slices"]
    edge_pos = np.minimum(summary["max_pos_slice"], summary["n_slices"] - 1
                          - summary["max_pos_slice"]) / summary["n_slices"]
    L.append(
        f"2. **A large share of the extreme slices are at the edge of the volume.** The "
        f"most negative slice lies in the outermost 10% of the stack for "
        f"**{int((edge < 0.10).sum())} of {len(summary)}** patients, and the most positive "
        f"slice for **{int((edge_pos < 0.10).sum())} of {len(summary)}**. Those slices are "
        f"body wall, air, or a small off-anatomy bright artefact — not liver."
    )
    L.append(
        "3. **Integrated Gradients needs a blur baseline here.** With a zero baseline the "
        "completeness residual does not converge at any practical step count "
        "(relative 1.24/4.48/5.46 at 64 steps, still 0.34/0.68/0.73 at 256, non-monotone): "
        "a uniform image sits in LayerNorm's near-singular region. The blur baseline "
        "converges monotonically to 0.010/0.017/0.034 at 256 steps, which is what §3 reports."
    )
    L.append(
        f"4. **On the resection cohort a large part of the decision magnitude comes from "
        f"slices that contain no anatomy at all** — a preprocessing artefact carrying "
        f"{summary.loc[res, 'degenerate_c_s_fraction'].min():.0%}–"
        f"{summary.loc[res, 'degenerate_c_s_fraction'].max():.0%} of `Σ|c_s|` there and "
        f"{summary.loc[sor, 'degenerate_c_s_fraction'].max():.1%} or less on Soramic. "
        f"Mechanism, quantification and cohort asymmetry are in Appendix C."
    )

    L += ["", "## 2. Method", ""]
    L.append(
        f"The patient embedding is the mean over **every** slice along axis "
        f"{enc['axes']} of the volume (`n_per_axis={enc['n_per_axis']}`), and all "
        f"ensemble members are linear, so with `z̄ = (1/S) Σ_s f(x_s)` and "
        f"`S(z) = (1/M) Σ_m σ(a_m(β_m·z + b_m) + c_m)` the local decision direction "
        f"`β_eff = ∇_z logit S(z̄)` yields an **exact** additive decomposition"
    )
    L += ["", "```", "c_s = β_eff · f(x_s) / S,        Σ_s c_s = β_eff · z̄", "```", ""]
    L.append(
        f"Which slices matter is therefore read off the model rather than chosen by hand. "
        f"The two figures below attribute at the voxel level within those slices: "
        f"**Integrated Gradients** ({meta_all['ig_steps']} midpoint steps, baseline "
        f"`{meta_all['baseline']}`) on the top-{meta_all['top_k_slices']} slices by "
        f"`|c_s|`, with the most positive and most negative slice forced in, and "
        f"**Gradient×Input** on every slice, stacked into a 3D volume and projected along "
        f"the slicing axis. Both are mapped back to the native voxel grid, inverting the "
        f"backbone's hidden 224→256 resize and 224 centre crop, and rescaled to preserve "
        f"their signed sum."
    )
    L.append("")
    L.append(
        "The per-slice profile `c_s` itself is plotted in Appendix A; the constant-input "
        "slices it exposes, and how they are excluded from the extreme-slice selection, "
        "are described in Appendix C."
    )

    L += ["", "## 3. Validation gates", ""]
    L.append(f"- Recomputed `z̄` vs the cached embedding the thesis tables use: "
             f"max **{summary['embedding_gate_max_abs'].max():.2e}** over "
             f"{len(summary)} patients")
    L.append(f"- Decomposition identity `Σ_s c_s` vs `β_eff·z̄`: max "
             f"**{summary['decomposition_gap'].max():.2e}**")
    L.append(f"- Head reconstruction (closed-form ensemble score vs `predict_proba`): "
             f"**{meta_all['head_reconstruction_max_residual']:.2e}** — the residual is "
             f"libsvm's iterative pairwise-coupling step in the Platt-scaled `L-SVM` "
             f"member, not the unwinding (same behaviour as the gene-side report)")
    L.append(f"- IG completeness `Σ IG − Δtarget`, each residual against its own slice's "
             f"target change: worst-patient median **"
             f"{summary['ig_relative_residual_median'].max():.3f}**, worst single slice "
             f"**{summary['ig_relative_residual'].max():.3f}** "
             f"(max absolute {summary['ig_max_abs_residual'].max():.2e})")

    L += ["", "## 4. Selected cases", ""]
    L.append(
        "Every patient is attributed with the head fit on all labelled resection "
        "patients, so `β_eff` differs between patients only through their own embedding. "
        "Resection `p` is in-sample (see the note at the top)."
        if meta_all.get("resection_head", "oof") == "full" else
        "Resection probabilities are **out-of-fold** and each resection patient is "
        "attributed with its own held-out fold's head; Soramic uses the head refit "
        "on all labelled resection patients."
    )
    L += [""] + _case_table(main)
    L += ["", "`Σc_s = β_eff·z̄` is the **linear part** of the score at that patient's "
          "operating point; the member intercepts carry the rest, so its sign need not "
          "track `p`. `max-pos`/`max-neg slice` are the extremes among slices that carry "
          "anatomy (Appendix C).", ""]

    L += ["## 5. Figures", ""]
    for cohort in cohorts:
        L += [f"### {cohort}", ""]
        L.append(f"![IG on the extreme slices]({figrel}/top_slices_{cohort}.png)")
        L.append("")
        L.append(f"![Gradient×Input MIP]({figrel}/saliency_mip_{cohort}.png)")
        L.append("")

    L += ["## 6. Caveats", ""]
    L += [
        f"- **Spatial resolution.** `{enc['model']}` gives a 7×7 patch grid over the "
        f"{enc['img_size']}px input; after the anisotropic resize of an elongated "
        f"sagittal slice one patch covers tens of millimetres. The attribution is "
        f"regional, not textural — which is why both figures are pooled to that grid.",
        f"- **Frozen backbone** (`freeze_backbone={enc['freeze_backbone']}`): only the "
        f"projection MLP was trained, so the spatial features are ImageNet's.",
        "- **Centre crop.** The backbone transform resizes 224→256 and centre-crops back "
        "to 224, so a ~6% border of every slice is never seen by the encoder and is given "
        "exactly zero attribution.",
        "- **`β_eff` is patient-specific** — the local gradient of a mean of sigmoids. "
        "`c_s` decomposes the linearised logit exactly, not `logit S` itself.",
        "- **Per-slice p99 normalisation** means the attribution is with respect to the "
        "model's input, not raw MRI intensity.",
        "- **The MIP discards the slice axis.** A positive and a negative peak on the same "
        "ray compete, and only the larger survives, so a blank region in the MIP is not "
        "evidence of no contribution. Colour scales are per-panel, so saturation is not "
        "comparable between patients.",
        "- **Deployed vs training input.** The encoder was trained on the raw resection "
        "volumes (resampled on load) but the deployed embeddings are extracted from the "
        "preprocessed root without resampling. The attribution follows the **deployed** "
        "path — the one behind every number in the thesis tables.",
        "",
    ]

    L += ["## 7. Regenerate", "", "```",
          f"python -m hcc_multimodal.interpretability.image_saliency \\",
          f"  --model-id {meta_all['model_id']} --members-csv {meta_all['members_csv']} \\",
          f"  --cohorts {' '.join(meta_all['cohorts'])} "
          f"--resection-head {meta_all.get('resection_head', 'oof')} "
          f"--extra-tp {meta_all.get('extra_tp', 0)} --extra-tn {meta_all.get('extra_tn', 0)} \\",
          f"  --top-k-slices {meta_all['top_k_slices']} --ig-steps {meta_all['ig_steps']} \\",
          f"  --output-dir {in_dir}",
          f"python -m hcc_multimodal.interpretability.image_saliency_plots \\",
          f"  --input-dir {in_dir} --fig-dir {fig_dir} --report {report}",
          "```", ""]

    # ------------------------------------------------------------------ appendices
    L += ["---", "", "## Appendix A — Per-slice contribution profiles", ""]
    L.append(
        "`c_s` against slice index for every case, main and extra. The tumour's extent "
        "along the slicing axis is shaded green and constant-input slices (Appendix C) "
        "grey. This is the figure that shows *how the score is distributed over the "
        "volume* before any voxel-level attribution: the two figures in §5 are zoom-ins "
        "on the extremes of these curves."
    )
    for cohort in cohorts:
        L += ["", f"### {cohort}", "",
              f"![Per-slice contribution profile]({figrel}/slice_profile_{cohort}.png)", ""]

    L += ["## Appendix B — Where the extreme slices sit relative to the tumour", ""]
    L.append(
        "Slice-level overlap only: whether the index of the extreme slice falls within "
        "the first and last slice containing tumour. No voxel-level masking is involved."
    )
    L += ["", "| Cohort | Case | SID | tumour slice extent | max-pos slice | in tumour? | "
          "max-neg slice | in tumour? |", "|---|---|---:|---:|---:|:-:|---:|:-:|"]
    for (_, r), (_, k) in zip(summary.iterrows(), pk.iterrows()):
        ext = f"{k['lo']}–{k['hi']}" if k["lo"] is not None else "—"
        mark = lambda b: "—" if b is None else ("**yes**" if b else "no")  # noqa: E731
        L.append(f"| {r['cohort']} | {r['case']} | {r['SID']} | {ext} | "
                 f"{r['max_pos_slice']} | {mark(k['pos_in'])} | "
                 f"{r['max_neg_slice']} | {mark(k['neg_in'])} |")
    L += ["",
          "Extremes are over slices carrying anatomy (constant-input slices excluded). "
          "`c_s` is signed, so a tumour slice can legitimately contribute negatively — "
          "the point of the column is *whether the model's strongest evidence sits on "
          "the lesion at all*.", ""]

    L += ["## Appendix C — Constant-input (degenerate) slices", ""]
    L.append(
        "`_normalize_slice` (`eval/data.py`, mirrored in `contrastive/data.py`) does "
        "`np.clip(s, 0, p99)`. When a slice's 99th percentile is **negative** — a "
        "background-only slice of a volume whose background is negative — `a_min > a_max`, "
        "so numpy returns `p99` at every pixel, and the following `if p99 > 0` rescale is "
        "skipped. The slice reaches the encoder as a **constant image at a large negative "
        "value** (~-54 in ImageNet-normalised units, against [-2.12, 2.64] for a real "
        "slice): far outside anything the backbone saw in training, carrying no anatomy, "
        "and still averaged into the patient embedding."
    )
    L += ["",
          "Such slices are flagged by the mechanism itself (`percentile(slice, 99) <= 0` "
          "on the source volume) rather than by testing the tensor for constancy, which "
          "is unreliable: two bilinear resizes leave float32 noise on the constant, and "
          "the ImageNet normalisation then gives each channel its own constant. They are "
          "excluded from the extreme-slice selection — a constant image has no spatial "
          "story to tell — but they remain inside the mean-pooled embedding, so their "
          "share of `Σ|c_s|` is reported here instead.", ""]
    L += ["| Cohort | Case | SID | slices | degenerate | share of Σ\\|c_s\\| |",
          "|---|---|---:|---:|---:|---:|"]
    for _, r in summary.iterrows():
        L.append(
            f"| {r['cohort']} | {r['case']} | {r['SID']} | {r['n_slices']} | "
            f"{r['n_degenerate']} ({r['degenerate_fraction']:.1%}) | "
            f"{r['degenerate_c_s_fraction']:.1%} |"
        )
    L += ["", "This is a property of the **deployed** pipeline, not of this analysis: every "
          "cached embedding and every AUROC in the thesis was produced with it. It is "
          "reported, not fixed — changing `_normalize_slice` would invalidate all of them.", ""]

    if len(extra):
        L += ["## Appendix D — Additional confident hits", ""]
        L.append(
            "The next-ranked true positives and true negatives by predicted probability. "
            "They are kept out of §5 so that each outcome × prediction category is "
            "represented there by a single exemplar, but they are what the pooled counts "
            "in finding 1 are based on: a pattern seen in one confident hit is not "
            "distinguishable from a coincidence."
        )
        L += [""] + _case_table(extra)
        for cohort in cohorts:
            if not len(extra[extra["cohort"] == cohort]):
                continue
            L += ["", f"### {cohort}", "",
                  f"![IG on the extreme slices]({figrel}/top_slices_{cohort}_extra.png)", "",
                  f"![Gradient×Input MIP]({figrel}/saliency_mip_{cohort}_extra.png)", ""]

    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(L))
    print(f"  wrote {report}")


def run(args) -> None:
    meta_all = json.loads((args.input_dir / "saliency_meta.json").read_text())
    meta = meta_all["encoder"]
    summary = pd.read_csv(args.input_dir / "saliency_summary.csv")
    summary = summary.sort_values(["cohort", "case_order"])

    is_main = summary["case"].isin(args.main_cases)
    main, extra = summary[is_main], summary[~is_main]
    if not len(main):
        raise SystemExit(f"--main-cases {args.main_cases} matched no rows; "
                         f"available: {sorted(summary['case'].unique())}")

    for cohort in summary["cohort"].unique():
        rows = summary[summary["cohort"] == cohort]
        m = main[main["cohort"] == cohort]
        e = extra[extra["cohort"] == cohort]
        print(f"{cohort}: {len(m)} main + {len(e)} appendix cases")
        fig_slice_profile(cohort, rows, args.input_dir, args.fig_dir)
        for subset, suffix in ((m, ""), (e, "_extra")):
            if not len(subset):
                continue
            fig_top_slices(cohort, subset, args.input_dir, args.fig_dir, meta,
                           args.mask_overlay, stem=f"top_slices_{cohort}{suffix}")
            fig_saliency_mip(cohort, subset, args.input_dir, args.fig_dir, meta,
                             args.mask_overlay, stem=f"saliency_mip_{cohort}{suffix}")

    if args.report is not None:
        write_report(summary, main, extra, meta_all, args.input_dir, args.fig_dir,
                     args.report)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", type=Path,
                   default=Path("results/eval/interpretability/image_saliency/d7085bf5"))
    p.add_argument("--fig-dir", type=Path, default=Path("reports/0810/image_saliency"))
    p.add_argument("--no-mask-overlay", dest="mask_overlay", action="store_false")
    p.add_argument("--main-cases", nargs="+", default=list(CASES),
                   help="cases carried by the main text's two voxel-level figures, one "
                        "exemplar per outcome x prediction category. Everything else "
                        "(the rank-2/3 confident hits) goes to Appendix D")
    p.add_argument("--report", type=Path, default=Path("reports/0810/0810_image_saliency.md"),
                   help="Markdown report to write; pass an empty string to skip")
    args = p.parse_args()
    if args.report is not None and str(args.report) in ("", "."):
        args.report = None
    return args


if __name__ == "__main__":
    run(parse_args())
