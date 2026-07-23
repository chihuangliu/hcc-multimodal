"""KM figures for the screen winners (Route A, k-means split, with CI bands).

Generates a (models × cohorts) panel for a set of selected models, computing
Route-A risk scores on the fly (no pre-saved parquets needed).

Run:  python -m hcc_multimodal.survival.screen_km_figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

from hcc_multimodal.eval.data import PROJECT_ROOT

from . import COHORTS
from .analysis import analyze_groups, concordance
from .cutoffs import kmeans_within
from .data import load_source_aligned
from .plots import km_panel
from .risk_scores import route_a_scores

DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "0620"

SCREEN_WINNERS = [
    ("06c598c0", "lr"),
    ("1361bef2", "rf"),
]


def _build_panel(models: list[tuple[str, str]]) -> tuple[dict, list[str]]:
    panel: dict = {}
    model_ids: list[str] = []
    for source, head in models:
        train = load_source_aligned(source, "resection")
        panel[source] = {}
        model_ids.append(source)
        for cohort in COHORTS:
            test = load_source_aligned(source, cohort)
            oof, test_scores = route_a_scores(head, train, test.X)
            groups, _ = kmeans_within(oof, train.rfs_2year, test_scores)
            stats = analyze_groups(groups, test.time, test.event)
            stats["c_index"] = concordance(test_scores, test.time, test.event)
            panel[source][cohort] = {
                "time": test.time,
                "event": test.event,
                "groups": groups,
                "stats": stats,
            }
    return panel, model_ids


def main(args: argparse.Namespace) -> None:
    panel, model_ids = _build_panel(SCREEN_WINNERS)
    out_path = args.report_dir / "km_screen_winners.png"
    km_panel(
        panel, model_ids, list(COHORTS), out_path,
        "Screen winners — Route A, k-means split",
        ci_show=True,
    )


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--report-dir", type=Path, default=DEFAULT_REPORT_DIR,
        help="Output dir (default: %(default)s)",
    )
    return p.parse_args()


if __name__ == "__main__":
    main(_parse())
