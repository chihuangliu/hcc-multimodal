"""Generate Kaplan-Meier curves with CI for specified model+head+cohort combos.

Uses 2-Year RFS classifier risk score with kmeans cutoff. Outputs PNG + SVG
figures and prints a statistics summary.

Example:
    python -m hcc_multimodal.survival.survival_km \
        --panels "9109a6c2:lr:soramic" "06c598c0:lr:soramic" "1361bef2:rf:lusanne" \
        --output reports/0620/km_5_2.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hcc_multimodal.survival.analysis import analyze_groups, concordance
from hcc_multimodal.survival.cutoffs import CUTOFF_METHODS
from hcc_multimodal.survival.data import load_source_aligned
from hcc_multimodal.survival.plots import _draw_subplot
from hcc_multimodal.survival.risk_scores import route_a_scores
from sklearn.metrics import roc_auc_score


def _auroc(scores: pd.Series, rfs_2year: pd.Series) -> float | None:
    y = rfs_2year.dropna()
    common = scores.index.intersection(y.index)
    y = y.loc[common].astype(int)
    if y.nunique() < 2:
        return None
    return float(roc_auc_score(y, scores.loc[common]))


def compute_panel(model_id: str, head: str, cohort: str, cutoff: str = "kmeans_frozen") -> dict:
    train = load_source_aligned(model_id, "resection")
    test = load_source_aligned(model_id, cohort)

    oof, test_scores = route_a_scores(head, train, test.X)
    groups, _ = CUTOFF_METHODS[cutoff](oof, train.rfs_2year, test_scores)

    stats = analyze_groups(groups, test.time, test.event)
    cidx = concordance(test_scores, test.time, test.event)
    auroc = _auroc(test_scores, test.rfs_2year)
    stats["c_index"] = cidx
    stats["auroc"] = auroc

    return {
        "time": test.time,
        "event": test.event,
        "groups": groups,
        "stats": stats,
        "model_id": model_id,
        "head": head,
        "cohort": cohort,
    }


def print_stats(panels: list[dict]) -> None:
    print("\n=== Statistics ===")
    for p in panels:
        s = p["stats"]
        print(f"\n{p['model_id']} ({p['head']}) on {p['cohort']}:")
        print(f"  n (hi/lo)       : {s['n_high']}/{s['n_low']}")
        print(f"  AUROC           : {s['auroc']:.3f}" if s.get("auroc") else "  AUROC           : —")
        print(f"  C-index         : {s['c_index']:.3f}" if s.get("c_index") else "  C-index         : —")

        hr = s.get("hr_high_vs_low")
        if hr is not None:
            print(f"  HR (95% CI)     : {hr:.2f} ({s['hr_ci_low']:.2f}–{s['hr_ci_high']:.2f})")
        else:
            print("  HR (95% CI)     : —")

        p_val = s.get("logrank_p")
        if p_val is not None:
            print(f"  log-rank p      : {p_val:.4f}" if p_val >= 1e-4 else f"  log-rank p      : {p_val:.2e}")
        else:
            print("  log-rank p      : —")

        for grp in ("high", "low"):
            g = s.get(grp)
            if g:
                med = f"{g['median_rfs']:.1f}" if g.get("median_rfs") is not None else "NR"
                r12 = f"{g['rfs_rate_12mo']:.1%}" if g.get("rfs_rate_12mo") is not None else "—"
                r24 = f"{g['rfs_rate_24mo']:.1%}" if g.get("rfs_rate_24mo") is not None else "—"
                print(f"  {grp:4s} — median RFS: {med} mo, 12-mo RFS: {r12}, 24-mo RFS: {r24}")


def make_figure(panels: list[dict], output: Path) -> None:
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.5), squeeze=False)
    for i, p in enumerate(panels):
        title = f"{p['model_id']} ({p['head'].upper()}) · {p['cohort'].capitalize()}"
        _draw_subplot(
            axes[0][i], p["time"], p["event"], p["groups"], p["stats"],
            title=title, ci_show=True,
        )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {output} (+ .svg)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--panels", nargs="+", required=True,
                   help="Each panel as 'model_id:head:cohort' (e.g. '9109a6c2:lr:soramic')")
    p.add_argument("--cutoff", default="kmeans_frozen", choices=list(CUTOFF_METHODS),
                   help="Cutoff strategy for the high/low split (default: kmeans_frozen)")
    p.add_argument("--output", type=Path, default=Path("reports/0620/km_5_2.png"),
                   help="Output figure path (PNG; SVG auto-generated)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    panels = []
    for spec in args.panels:
        model_id, head, cohort = spec.split(":")
        print(f"Computing: {model_id} ({head}) on {cohort} [{args.cutoff}]...")
        panels.append(compute_panel(model_id, head, cohort, args.cutoff))

    print_stats(panels)
    make_figure(panels, args.output)
