"""Task 5 — AUROC vs C-index scatter per cohort (editable SVG + PNG).

Visualises the screen: each point is a source (best head by AUROC) at its 2-year
AUROC (x) and survival C-index (y). The radiomic baseline is marked, and dashed
crosshairs at the baseline split the plane — the upper-right quadrant beats the
baseline on both axes. Points with a significant within-median log-rank
(p < 0.05) are highlighted.

Writes reports/0620/screen_auc_cindex.{png,svg}.

Run:  python -m hcc_multimodal.survival.screen_figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from hcc_multimodal.eval.data import PROJECT_ROOT  # noqa: E402

from . import COHORTS  # noqa: E402

SCREEN_CSV = PROJECT_ROOT / "results" / "eval" / "survival" / "screen.csv"
DEFAULT_OUT = PROJECT_ROOT / "reports" / "0620" / "screen_auc_cindex.png"

_LABEL = {"soramic": "Soramic", "lusanne": "Lausanne"}


def _best_head(df: pd.DataFrame, cohort: str) -> pd.DataFrame:
    # AUROC / C-index are split-independent; significance uses the k-means log-rank.
    sub = df[(df.cohort == cohort) & (df.split == "kmeans_within")].copy()
    return sub.loc[sub.groupby("source")["auroc"].idxmax()]


def _draw(ax, data: pd.DataFrame, cohort: str) -> None:
    base = data[data.source == "radiomic"].iloc[0]
    ax.axvline(base["auroc"], ls="--", color="0.6", lw=1)
    ax.axhline(base["c_index"], ls="--", color="0.6", lw=1)
    ax.axhline(0.5, ls=":", color="0.8", lw=1)

    for _, r in data.iterrows():
        if r.source == "radiomic":
            ax.scatter(r["auroc"], r["c_index"], marker="*", s=240, color="#c44e52",
                       zorder=5, edgecolor="k", linewidth=0.5)
            ax.annotate("radiomic", (r["auroc"], r["c_index"]),
                        textcoords="offset points", xytext=(6, 6), fontsize=8, color="#c44e52")
            continue
        sig = pd.notna(r["logrank_p"]) and r["logrank_p"] < 0.05
        dual = r["auroc"] > base["auroc"] and r["c_index"] > base["c_index"]
        color = "#2ca02c" if sig else ("#4c72b0" if dual else "0.5")
        ax.scatter(r["auroc"], r["c_index"], s=55, color=color, zorder=4,
                   edgecolor="k", linewidth=0.3)
        if sig or r["auroc"] > base["auroc"] + 0.06:
            ax.annotate(r["source"], (r["auroc"], r["c_index"]),
                        textcoords="offset points", xytext=(5, -3), fontsize=7)

    ax.set_title(f"{_LABEL[cohort]}  (baseline: AUROC={base['auroc']:.3f}, C={base['c_index']:.3f})",
                 fontsize=11)
    ax.set_xlabel("2-year RFS AUROC")
    ax.set_ylabel("Survival C-index")


def main(args: argparse.Namespace) -> None:
    df = pd.read_csv(SCREEN_CSV)
    fig, axes = plt.subplots(1, len(COHORTS), figsize=(7.0 * len(COHORTS), 5.5), squeeze=False)
    for j, cohort in enumerate(COHORTS):
        _draw(axes[0][j], _best_head(df, cohort), cohort)
    # shared legend
    handles = [
        plt.Line2D([], [], marker="*", color="#c44e52", ls="", ms=14, mec="k", label="radiomic baseline"),
        plt.Line2D([], [], marker="o", color="#2ca02c", ls="", ms=8, mec="k", label="significant k-means log-rank (p<0.05)"),
        plt.Line2D([], [], marker="o", color="#4c72b0", ls="", ms=8, mec="k", label="beats baseline on both axes"),
        plt.Line2D([], [], marker="o", color="0.5", ls="", ms=8, mec="k", label="other"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9, frameon=False)
    fig.suptitle("Model screen — 2-year AUROC vs survival C-index (best head per model)", fontsize=13)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {args.out} (+ .svg)")


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output path (default: %(default)s)")
    return p.parse_args()


if __name__ == "__main__":
    main(_parse())
