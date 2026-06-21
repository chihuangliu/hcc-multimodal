"""Task 2 — stratify and run survival statistics.

For each (model, route, cohort) the cutoff-free C-index is computed once, then
four cutoff strategies (see :mod:`cutoffs`) are applied. Split counts are always
recorded. Full survival stats (KM medians, log-rank, Cox HR) are computed for the
within-cohort-median (primary) and frozen-resection-median (sensitivity)
strategies always, and for k-means / Youden only when their groups are balanced
(min 5 per arm); otherwise only their split counts are reported.

Writes a combined summary under results/eval/survival/.

Run:  python -m hcc_multimodal.survival.run_analysis
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime

import pandas as pd

from hcc_multimodal.eval.data import PROJECT_ROOT

from . import COHORTS, MODELS, ROUTES
from .analysis import analyze_groups, concordance, is_balanced, route_agreement
from .cutoffs import CUTOFF_METHODS, PRIMARY_METHOD
from .data import load_survival_outcomes

SURVIVAL_DIR = PROJECT_ROOT / "results" / "eval" / "survival"
SCORES_DIR = SURVIVAL_DIR / "scores"

# Methods that always get full survival stats regardless of balance.
ALWAYS_FULL = {"median_within", "median_frozen"}


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def main(args: argparse.Namespace) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    resection_labels = load_survival_outcomes("resection")["rfs_2year"]
    summary = {"git_commit": _git_commit(), "timestamp": datetime.now().isoformat(), "results": {}}

    for model_id, head in MODELS.items():
        summary["results"][model_id] = {"head": head, "routes": {}, "agreement": {}}
        primary_groups: dict[str, dict[str, pd.Series]] = {r: {} for r in ROUTES}

        for route in ROUTES:
            df = pd.read_parquet(SCORES_DIR / f"{model_id}_{route}.parquet")
            train = df[df.split == "train"].set_index("SID")["risk_score"]
            train_labels = resection_labels.reindex(train.index)
            route_out = {"cohorts": {}}

            for cohort in COHORTS:
                sub = df[(df.split == "test") & (df.cohort == cohort)].set_index("SID")
                scores, time, event = sub["risk_score"], sub["time"], sub["event"].astype(int)
                cell = {"c_index": concordance(scores, time, event), "methods": {}}

                for name, fn in CUTOFF_METHODS.items():
                    groups, info = fn(train, train_labels, scores)
                    n_high = int((groups == "high").sum())
                    n_low = int((groups == "low").sum())
                    balanced = is_balanced(n_high, n_low)
                    entry = {**info, "n_high": n_high, "n_low": n_low, "balanced": balanced}
                    if name in ALWAYS_FULL or balanced:
                        entry["survival"] = analyze_groups(groups, time, event)
                    else:
                        entry["survival"] = None  # too imbalanced: counts only
                    cell["methods"][name] = entry
                    if name == PRIMARY_METHOD:
                        primary_groups[route][cohort] = groups

                route_out["cohorts"][cohort] = cell
                _print_cell(model_id, route, cohort, cell)
            summary["results"][model_id]["routes"][route] = route_out

        for cohort in COHORTS:
            summary["results"][model_id]["agreement"][cohort] = route_agreement(
                primary_groups["a"][cohort], primary_groups["b"][cohort]
            )

    SURVIVAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SURVIVAL_DIR / f"survival_summary_{timestamp}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary → {out_path}")


def _print_cell(model_id: str, route: str, cohort: str, cell: dict) -> None:
    c = cell["c_index"]
    print(f"{model_id} {route.upper()} {cohort:8s} | C-index={c:.3f}" if c is not None else
          f"{model_id} {route.upper()} {cohort:8s} | C-index=n/a")
    for name, e in cell["methods"].items():
        s = e["survival"]
        if s is None:
            print(f"    {name:14s} hi/lo={e['n_high']:>3}/{e['n_low']:<3} | (imbalanced → counts only)")
        else:
            hr = s["hr_high_vs_low"]
            p = s["logrank_p"]
            hr_s = f"{hr:.2f}" if hr is not None else "n/a"
            p_s = f"{p:.3f}" if p is not None else "n/a"
            print(f"    {name:14s} hi/lo={e['n_high']:>3}/{e['n_low']:<3} | HR={hr_s:>5} | logrank p={p_s}")


def _parse() -> argparse.Namespace:
    return argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    ).parse_args()


if __name__ == "__main__":
    main(_parse())
