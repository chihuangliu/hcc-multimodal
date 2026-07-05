"""Survival stratification for the best benchmark-grid heads (0629-style).

Selects the top-N (feature-selection, classifier) combinations by **Soramic**
transfer AUROC (from ``results/eval/grid/grid_transfer_soramic.csv``), scores each
with :func:`hcc_multimodal.survival.grid_scores.route_grid_scores`, and applies the
resection-frozen cutoff strategies to produce a per-combo × per-cutoff survival table
on Soramic (AUROC, C-index, n hi/lo, HR (CI), log-rank p, median RFS) — the same shape
as ``reports/0629/0629_survical_analysis_v2.md`` §5.1.1.

The single best (fs, model, cutoff) on Soramic (lowest log-rank p among balanced,
correct-direction splits) is then **applied to Lausanne** as a held-out generalization
check, and KM curves for both cohorts are drawn.

Example
-------
    python -m hcc_multimodal.survival.survival_stratify_grid --top 5 --km
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score

from hcc_multimodal.eval.grid import SELECT_K_DEFAULT
from hcc_multimodal.survival.analysis import analyze_groups, concordance, is_balanced
from hcc_multimodal.survival.cutoffs import CUTOFF_METHODS
from hcc_multimodal.survival.data import load_source_aligned
from hcc_multimodal.survival.grid_scores import route_grid_scores
from hcc_multimodal.survival.plots import _draw_subplot

matplotlib.rcParams["svg.fonttype"] = "none"

DEFAULT_CUTOFFS = ["median_frozen", "kmeans_frozen", "kmeans_log_frozen", "youden_frozen"]


def _labelled_auroc(scores: pd.Series, rfs_2year: pd.Series) -> float | None:
    mask = rfs_2year.notna()
    common = scores.index.intersection(rfs_2year[mask].index)
    if len(common) < 5:
        return None
    y = rfs_2year.loc[common].astype(int)
    if y.nunique() < 2:
        return None
    return float(roc_auc_score(y, scores.loc[common]))


def _stratify_rows(fs, model, oof, test_scores, train, test, cohort, cutoffs):
    """One row per cutoff for a (fs, model) combo on a cohort."""
    auroc = _labelled_auroc(test_scores, test.rfs_2year)
    cidx = concordance(test_scores, test.time, test.event)
    rows = []
    for cutoff in cutoffs:
        groups, _ = CUTOFF_METHODS[cutoff](oof, train.rfs_2year, test_scores)
        n_high = int((groups == "high").sum())
        n_low = int((groups == "low").sum())
        balanced = is_balanced(n_high, n_low)
        row = {
            "model": model, "fs": fs, "cohort": cohort, "cutoff": cutoff,
            "auroc": auroc, "c_index": cidx,
            "n_high": n_high, "n_low": n_low, "balanced": balanced,
            "hr": None, "hr_lo": None, "hr_hi": None, "logrank_p": None,
            "median_rfs_high": None, "median_rfs_low": None,
        }
        if balanced:
            stats = analyze_groups(groups, test.time, test.event)
            row.update({
                "hr": stats["hr_high_vs_low"],
                "hr_lo": stats["hr_ci_low"], "hr_hi": stats["hr_ci_high"],
                "logrank_p": stats["logrank_p"],
                "median_rfs_high": (stats["high"] or {}).get("median_rfs"),
                "median_rfs_low": (stats["low"] or {}).get("median_rfs"),
            })
        rows.append(row)
    return rows


def _select_best(df: pd.DataFrame) -> pd.Series:
    """Best row: balanced, correct-direction (HR>1), lowest log-rank p."""
    cand = df[df["balanced"] & df["logrank_p"].notna()].copy()
    correct = cand[cand["hr"] > 1.0]
    pool = correct if len(correct) else cand
    if len(pool) == 0:
        return df.iloc[0]
    return pool.sort_values("logrank_p").iloc[0]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-id", default="9109a6c2")
    p.add_argument("--select-k", type=int, default=SELECT_K_DEFAULT)
    p.add_argument("--top", type=int, default=5)
    p.add_argument("--primary", default="soramic")
    p.add_argument("--confirm", default="lusanne")
    p.add_argument("--transfer-csv", type=Path,
                   default=Path("results/eval/grid/grid_transfer_soramic.csv"))
    p.add_argument("--cutoffs", nargs="+", default=DEFAULT_CUTOFFS)
    p.add_argument("--output-dir", type=Path, default=Path("results/eval/survival"))
    p.add_argument("--fig-dir", type=Path, default=Path("reports/0706"))
    p.add_argument("--km", action="store_true", help="draw KM curves for the best combo")
    p.add_argument("--time-col", default="RFS_central",
                   help="time-to-event column (e.g. TTR_central for time-to-recurrence)")
    p.add_argument("--event-col", default="RFS_central_event",
                   help="event-indicator column matching --time-col")
    p.add_argument("--tag", default="",
                   help="suffix for output CSV/figure names, e.g. 'ttr_central' "
                        "(default: RFS endpoint, no suffix, matching the 0706 report)")
    return p.parse_args()


def main():
    args = parse_args()
    transfer = pd.read_csv(args.transfer_csv)
    top = transfer.sort_values("auroc", ascending=False).head(args.top)[["fs", "model"]]
    combos = list(top.itertuples(index=False, name=None))  # (fs, model)
    print(f"Top {args.top} combos by {args.primary} AUROC: "
          + ", ".join(f"{m}/{fs}" for fs, m in combos))

    ep = {"time_col": args.time_col, "event_col": args.event_col}
    train = load_source_aligned(args.model_id, "resection", **ep)
    primary = load_source_aligned(args.model_id, args.primary, **ep)
    confirm = load_source_aligned(args.model_id, args.confirm, **ep)

    # Cache the scores per combo so the confirm cohort reuses the refit head.
    scored: dict[tuple, dict] = {}
    rows = []
    for fs, model in combos:
        oof_p, sc_p, bp = route_grid_scores(fs, model, train, primary.X, args.select_k)
        oof_c, sc_c, _ = route_grid_scores(fs, model, train, confirm.X, args.select_k)
        scored[(fs, model)] = {
            "oof": oof_p, "primary_scores": sc_p, "confirm_scores": sc_c,
            "confirm_oof": oof_c, "best_params": bp,
        }
        rows += _stratify_rows(fs, model, oof_p, sc_p, train, primary, args.primary, args.cutoffs)
        print(f"  scored {model}/{fs}  (params={bp})")

    suffix = f"_{args.tag}" if args.tag else ""
    primary_df = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    primary_path = args.output_dir / f"grid_stratify_{args.primary}{suffix}.csv"
    primary_df.to_csv(primary_path, index=False)

    best = _select_best(primary_df)
    print(f"\nBest on {args.primary}: {best['model']}/{best['fs']} + {best['cutoff']} "
          f"(log-rank p={best['logrank_p']}, HR={best['hr']})")

    # Apply the best (fs, model) to the confirm cohort across all cutoffs.
    fs_b, model_b = best["fs"], best["model"]
    sc = scored[(fs_b, model_b)]
    confirm_rows = _stratify_rows(
        fs_b, model_b, sc["confirm_oof"], sc["confirm_scores"],
        train, confirm, args.confirm, args.cutoffs,
    )
    confirm_df = pd.DataFrame(confirm_rows)
    confirm_df["selected_cutoff"] = best["cutoff"]
    confirm_path = args.output_dir / f"grid_stratify_{args.confirm}{suffix}.csv"
    confirm_df.to_csv(confirm_path, index=False)
    print(f"Wrote {primary_path} and {confirm_path}")

    if args.km:
        _draw_km(best, scored, train, primary, confirm, args, suffix)


def _draw_km(best, scored, train, primary, confirm, args, suffix=""):
    fs_b, model_b, cutoff = best["fs"], best["model"], best["cutoff"]
    sc = scored[(fs_b, model_b)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), squeeze=False)
    for ax, cohort, cd, scores, oof in [
        (axes[0][0], args.primary, primary, sc["primary_scores"], sc["oof"]),
        (axes[0][1], args.confirm, confirm, sc["confirm_scores"], sc["confirm_oof"]),
    ]:
        groups, _ = CUTOFF_METHODS[cutoff](oof, train.rfs_2year, scores)
        stats = analyze_groups(groups, cd.time, cd.event)
        stats["c_index"] = concordance(scores, cd.time, cd.event)
        ylabel = "Time to recurrence (event-free)" if "TTR" in args.time_col else "Recurrence-free survival"
        _draw_subplot(ax, cd.time, cd.event, groups, stats,
                      title=f"{model_b}/{fs_b} · {cohort}", ci_show=True, ylabel=ylabel)
    fig.suptitle(f"Best grid head — {model_b}/{fs_b} + {cutoff}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    out = args.fig_dir / f"km_grid_best{suffix}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved KM → {out} (+ .svg)")


if __name__ == "__main__":
    main()
