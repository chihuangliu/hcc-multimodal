"""Scatter plots: in-CV AUC (n=all inference) vs ablation AUROC.

Four subplots (2×2):
  1. Slice-split CV   vs slice-split ablation   (9 models)
  2. Patient-split CV vs patient-split ablation  (8 models)
  3. Patient-split CV vs slice-split ablation    (8 matched pairs)
  4. Slice-split CV   vs patient-split ablation  (8 matched pairs)

CV AUC source: results/multimodal_prediction/embeddings_{id}_*nall*/summary.csv
               (best of LR / RF)
Ablation source: results/eval/ablation/embedding_{id}_rfs_2year_*.json
                 (latest timestamp, best of LR / RF)

Output: reports/0601/cv_ablation_scatter.png  (or --out <path>)
"""

import argparse
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
MULTIMODAL_DIR = ROOT / "results" / "multimodal_prediction"
ABLATION_DIR = ROOT / "results" / "eval" / "ablation"
DEFAULT_OUT = ROOT / "reports" / "0601" / "cv_ablation_scatter.png"

# (short label, training split, display group index)
MODEL_META: dict[str, tuple[str, str, int]] = {
    "6a1a1bdf": ("raw λ=0.1 n=10",        "slice",   1),
    "1361bef2": ("raw λ=0.1 n=10",        "patient", 1),
    "982a6fa2": ("raw λ=0.0 n=10",        "slice",   1),
    "a6f970d6": ("raw λ=0.0 n=10",        "patient", 1),
    "12e4ba6a": ("predef genes",           "slice",   2),
    "34e6806f": ("predef genes",           "patient", 2),
    "5d04e6ba": ("2y_before_cv genes",     "slice",   2),
    "9109a6c2": ("2y_before_cv genes",     "patient", 2),
    "dc7e1d10": ("raw λ=0.1 frozen",       "slice",   3),
    "5e3f71a0": ("raw λ=0.1 frozen",       "patient", 3),
    "a64b245f": ("raw λ=0.0 frozen",       "slice",   3),
    "06c598c0": ("raw λ=0.0 frozen",       "patient", 3),
    "050d401d": ("bbox λ=0.1 n=10",        "slice",   4),
    "f8aabb75": ("bbox λ=0.1 n=10",        "patient", 4),
    "e12b0592": ("bbox λ=0.0 n=10",        "slice",   4),
    "8715461c": ("bbox λ=0.0 n=10",        "patient", 4),
    "92b9afed": ("bbox λ=0.1 frozen",      "slice",   4),
}

# (slice_model_id, patient_model_id) sharing the same training config
PAIRS: list[tuple[str, str]] = [
    ("6a1a1bdf", "1361bef2"),
    ("982a6fa2", "a6f970d6"),
    ("12e4ba6a", "34e6806f"),
    ("5d04e6ba", "9109a6c2"),
    ("dc7e1d10", "5e3f71a0"),
    ("a64b245f", "06c598c0"),
    ("050d401d", "f8aabb75"),
    ("e12b0592", "8715461c"),
]

GROUP_COLORS = {1: "#4e79a7", 2: "#f28e2b", 3: "#59a14f", 4: "#e15759"}
GROUP_LABELS = {
    1: "G1 Raw λ sweep",
    2: "G2 Gene ablation",
    3: "G3 Frozen n=all",
    4: "G4 BBox",
}


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_cv(model_id: str) -> tuple[float, float]:
    """Best (AUC mean, AUC std) over LR/RF from the n=all CV summary.csv."""
    pattern = str(
        MULTIMODAL_DIR / f"embeddings_{model_id}_rfs_2year_in_cv_k100_raw*nall*" / "summary.csv"
    )
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No n=all CV result for {model_id}. Run multimodal_prediction.py first.")
    df = pd.read_csv(paths[-1])
    best_row = df.loc[df["AUC mean"].idxmax()]
    return float(best_row["AUC mean"]), float(best_row["AUC std"])


def _load_ablation(model_id: str) -> float:
    """Best AUROC (max over LR/RF) from the latest ablation JSON."""
    pattern = str(ABLATION_DIR / f"embedding_{model_id}_rfs_2year_*.json")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No ablation result for {model_id}.")
    with open(paths[-1]) as fh:
        data = json.load(fh)
    inner = data["results"]["average"].get("embedding") or data["results"]["average"].get("radiomic")
    lr_auc = inner.get("lr", {}).get("auroc", 0.0)
    rf_auc = inner.get("rf", {}).get("auroc", 0.0)
    return max(lr_auc, rf_auc)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _axis_limits(values: list[float], pad: float = 0.04) -> tuple[float, float]:
    lo, hi = min(values), max(values)
    rng = hi - lo or 0.1
    return lo - pad * rng, hi + pad * rng


def _draw_panel(
    ax: plt.Axes,
    xs: list[float],
    ys: list[float],
    labels: list[str],
    groups: list[int],
    title: str,
    xlabel: str,
    ylabel: str,
    annotate: bool = True,
) -> None:
    ax.set_title(title, fontsize=9, fontweight="bold", pad=5)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.2, linewidth=0.6)

    ax.set_xlim(_axis_limits(xs, pad=0.08))
    ax.set_ylim(_axis_limits(ys, pad=0.08))

    for x, y, lbl, g in zip(xs, ys, labels, groups):
        color = GROUP_COLORS[g]
        ax.plot(x, y, "o", color=color, ms=6, alpha=0.9, zorder=3)
        if annotate:
            ax.annotate(
                lbl, (x, y),
                textcoords="offset points", xytext=(5, 2),
                fontsize=5.5, color=color, alpha=0.85,
            )

    if len(xs) >= 3:
        r, p = stats.pearsonr(xs, ys)
        pstr = f"p={p:.3f}" if p >= 0.001 else "p<0.001"
        ax.text(
            0.04, 0.96, f"r = {r:+.2f},  {pstr}",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.85),
        )
        # Fitted regression line
        xarr = np.array(xs)
        slope, intercept = np.polyfit(xarr, ys, 1)
        x0, x1 = ax.get_xlim()
        ax.plot([x0, x1], [slope * x0 + intercept, slope * x1 + intercept],
                "k-", lw=1.0, alpha=0.5, zorder=1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(out: Path) -> None:
    cv: dict[str, tuple[float, float]] = {}
    abl: dict[str, float] = {}
    for mid in MODEL_META:
        cv[mid] = _load_cv(mid)
        abl[mid] = _load_ablation(mid)

    slice_ids   = [m for m, (_, s, _) in MODEL_META.items() if s == "slice"]
    patient_ids = [m for m, (_, s, _) in MODEL_META.items() if s == "patient"]
    slice_pair  = [p[0] for p in PAIRS]   # slice model of each matched pair
    patient_pair = [p[1] for p in PAIRS]  # patient model of each matched pair

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle("In-CV AUC (n=all) vs Ablation AUROC — 2-year RFS", fontsize=12, fontweight="bold", y=0.995)

    # 1. Slice CV vs slice ablation
    _draw_panel(
        axes[0, 0],
        xs=[cv[m][0] for m in slice_ids],
        ys=[abl[m]   for m in slice_ids],
        labels=[MODEL_META[m][0] for m in slice_ids],
        groups=[MODEL_META[m][2] for m in slice_ids],
        title="1. Slice-split CV vs Slice-split Ablation",
        xlabel="In-CV AUC  (slice split, n=all)",
        ylabel="Ablation AUROC  (slice-split model)",
    )

    # 2. Patient CV vs patient ablation
    _draw_panel(
        axes[0, 1],
        xs=[cv[m][0] for m in patient_ids],
        ys=[abl[m]   for m in patient_ids],
        labels=[MODEL_META[m][0] for m in patient_ids],
        groups=[MODEL_META[m][2] for m in patient_ids],
        title="2. Patient-split CV vs Patient-split Ablation",
        xlabel="In-CV AUC  (patient split, n=all)",
        ylabel="Ablation AUROC  (patient-split model)",
    )

    # 3. Patient CV vs slice ablation  (matched pairs)
    _draw_panel(
        axes[1, 0],
        xs=[cv[m][0]  for m in patient_pair],
        ys=[abl[m]    for m in slice_pair],
        labels=[MODEL_META[m][0] for m in patient_pair],
        groups=[MODEL_META[m][2] for m in patient_pair],
        title="3. Patient-split CV vs Slice-split Ablation  (matched pairs)",
        xlabel="In-CV AUC  (patient split, n=all)",
        ylabel="Ablation AUROC  (slice-split counterpart)",
    )

    # 4. Slice CV vs patient ablation  (matched pairs)
    _draw_panel(
        axes[1, 1],
        xs=[cv[m][0]  for m in slice_pair],
        ys=[abl[m]    for m in patient_pair],
        labels=[MODEL_META[m][0] for m in slice_pair],
        groups=[MODEL_META[m][2] for m in slice_pair],
        title="4. Slice-split CV vs Patient-split Ablation  (matched pairs)",
        xlabel="In-CV AUC  (slice split, n=all)",
        ylabel="Ablation AUROC  (patient-split counterpart)",
    )

    # Shared legend
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=GROUP_COLORS[g],
               markersize=8, label=GROUP_LABELS[g])
        for g in sorted(GROUP_COLORS)
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center", ncol=4, fontsize=8,
        framealpha=0.9, edgecolor="0.7",
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.text(
        0.5, -0.01,
        "Solid line = OLS regression fit.",
        ha="center", fontsize=7.5, color="0.4",
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.995])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")

    # Print Pearson r summary to stdout
    print("\nPearson r summary:")
    for title, xs, ys in [
        ("Slice CV vs slice ablation",       [cv[m][0] for m in slice_ids],   [abl[m] for m in slice_ids]),
        ("Patient CV vs patient ablation",   [cv[m][0] for m in patient_ids], [abl[m] for m in patient_ids]),
        ("Patient CV vs slice ablation",     [cv[m][0] for m in patient_pair],[abl[m] for m in slice_pair]),
        ("Slice CV vs patient ablation",     [cv[m][0] for m in slice_pair],  [abl[m] for m in patient_pair]),
    ]:
        r, p = stats.pearsonr(xs, ys)
        pstr = f"p={p:.3f}" if p >= 0.001 else "p<0.001"
        print(f"  {title:<42}  r={r:+.3f}  {pstr}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output figure path (default: %(default)s)")
    args = parser.parse_args()
    main(args.out)
