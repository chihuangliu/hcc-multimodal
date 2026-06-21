"""Task 3 — render Kaplan-Meier panel figures (editable SVG + PNG).

One figure per route: a (models x cohorts) grid of KM curves (high vs low risk)
using the primary cutoff (within-cohort median), each subplot annotated with the
log-rank p, Cox HR, and C-index.

Writes reports/{date}/km_route_{a,b}.{png,svg}.

Run:  python -m hcc_multimodal.survival.make_figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from hcc_multimodal.eval.data import PROJECT_ROOT

from . import COHORTS, MODELS
from .analysis import analyze_groups, concordance
from .cutoffs import CUTOFF_METHODS, PRIMARY_METHOD
from .plots import km_panel

SCORES_DIR = PROJECT_ROOT / "results" / "eval" / "survival" / "scores"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "0620"

# (route, cutoff method, output stem, figure title).
FIGURES = [
    ("a", PRIMARY_METHOD, "km_route_a",
     "Route A — 2-year RFS classifier probability (within-cohort median split)"),
    ("b", PRIMARY_METHOD, "km_route_b",
     "Route B — penalized Cox on PCA embeddings (within-cohort median split)"),
    ("a", "kmeans_within", "km_route_a_kmeans",
     "Route A — 2-year RFS classifier probability (k-means split)"),
    ("b", "kmeans_log_within", "km_route_b_kmeans_log",
     "Route B — penalized Cox on PCA embeddings (log k-means split)"),
]


def _build_panel(route: str, method: str) -> dict:
    panel: dict = {}
    for model_id in MODELS:
        df = pd.read_parquet(SCORES_DIR / f"{model_id}_{route}.parquet")
        train = df[df.split == "train"].set_index("SID")["risk_score"]
        panel[model_id] = {}
        for cohort in COHORTS:
            sub = df[(df.split == "test") & (df.cohort == cohort)].set_index("SID")
            scores, time, event = sub["risk_score"], sub["time"], sub["event"].astype(int)
            groups, _ = CUTOFF_METHODS[method](train, None, scores)
            stats = analyze_groups(groups, time, event)
            stats["c_index"] = concordance(scores, time, event)
            panel[model_id][cohort] = {
                "time": time, "event": event, "groups": groups, "stats": stats,
            }
    return panel


def main(args: argparse.Namespace) -> None:
    for route, method, stem, title in FIGURES:
        panel = _build_panel(route, method)
        out_path = args.report_dir / f"{stem}.png"
        km_panel(panel, list(MODELS), list(COHORTS), out_path, title)


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR, help="Output dir (default: %(default)s)")
    return p.parse_args()


if __name__ == "__main__":
    main(_parse())
