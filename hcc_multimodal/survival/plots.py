"""Kaplan-Meier panel figures, saved as editable SVG (+ PNG).

``svg.fonttype='none'`` keeps text as real glyphs (editable in Illustrator/Inkscape)
rather than outlined paths, matching the repo's other figure scripts.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from lifelines import KaplanMeierFitter  # noqa: E402

_COLORS = {"high": "#c44e52", "low": "#4c72b0"}


def _annotation(stats: dict) -> str:
    lines = []
    p = stats.get("logrank_p")
    if p is not None:
        lines.append(f"log-rank p = {p:.3f}" if p >= 1e-3 else "log-rank p < 0.001")
    hr = stats.get("hr_high_vs_low")
    if hr is not None:
        lines.append(f"HR = {hr:.2f} ({stats['hr_ci_low']:.2f}–{stats['hr_ci_high']:.2f})")
    c = stats.get("c_index")
    if c is not None:
        lines.append(f"C-index = {c:.3f}")
    return "\n".join(lines)


def _draw_subplot(ax, time: pd.Series, event: pd.Series, groups: pd.Series, stats: dict, title: str, *, ci_show: bool = False):
    for grp in ("low", "high"):
        mask = groups == grp
        if mask.sum() == 0:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(time[mask], event[mask], label=f"{grp} (n={int(mask.sum())})")
        kmf.plot_survival_function(ax=ax, color=_COLORS[grp], ci_show=ci_show, ci_alpha=0.15)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Months")
    ax.set_ylabel("Recurrence-free survival")
    ax.set_ylim(-0.02, 1.02)
    ax.text(
        0.98, 0.97, _annotation(stats), transform=ax.transAxes,
        ha="right", va="top", fontsize=8,
        bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85),
    )
    ax.legend(loc="lower left", fontsize=8)


def km_panel(panel: dict, models: list[str], cohorts: list[str], out_path: Path, suptitle: str, *, ci_show: bool = False):
    """Render a (models x cohorts) grid of KM plots.

    ``panel[model_id][cohort]`` -> dict(time, event, groups, stats).
    """
    nrows, ncols = len(models), len(cohorts)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.2 * nrows), squeeze=False)
    for i, model_id in enumerate(models):
        for j, cohort in enumerate(cohorts):
            cell = panel[model_id][cohort]
            _draw_subplot(
                axes[i][j], cell["time"], cell["event"], cell["groups"], cell["stats"],
                title=f"{model_id} · {cohort}", ci_show=ci_show,
            )
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path} (+ .svg)")
