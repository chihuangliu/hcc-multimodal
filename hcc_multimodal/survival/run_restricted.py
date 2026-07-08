"""Restricted-time (horizon-τ) survival report for the best benchmark-grid head.

Reuses the exact head-selection of :mod:`hcc_multimodal.survival.survival_stratify_grid` — top-N
(feature-selection, classifier) combos by Soramic transfer AUROC, scored with
:func:`grid_scores.route_grid_scores`, best head+cutoff chosen by full-follow-up
log-rank among balanced HR>1 splits — then, holding that head+cutoff fixed, runs the
restricted-time analysis (:mod:`restricted`) at each horizon τ on the primary
(Soramic) and confirm (Lausanne) cohorts:

  A. administrative-censoring KM/log-rank/HR/C-index at τ, and
  B. RMST(τ) per arm + ΔRMST(95% CI) + point-in-time survival-difference p.

RFS endpoint only (``RFS_central`` / ``RFS_central_event``). Writes one long-form
CSV per cohort (cohort × τ) and a KM figure per cohort with the τ horizons marked.

Example
-------
    python -m hcc_multimodal.survival.run_restricted --top 5 --km
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from hcc_multimodal.eval.grid import SELECT_K_DEFAULT
from hcc_multimodal.survival.analysis import analyze_groups, concordance, is_balanced
from hcc_multimodal.survival.cutoffs import CUTOFF_METHODS
from hcc_multimodal.survival.data import load_source_aligned
from hcc_multimodal.survival.grid_scores import route_grid_scores
from hcc_multimodal.survival.plots import _draw_subplot
from hcc_multimodal.survival.restricted import restricted_stats

matplotlib.rcParams["svg.fonttype"] = "none"

DEFAULT_CUTOFFS = ["median_frozen", "kmeans_frozen", "kmeans_log_frozen", "youden_frozen"]
DEFAULT_TAUS = [12.0, 24.0, 48.0]


def _select_best_head(combos, train, primary, select_k, cutoffs):
    """Full-follow-up stratify table → best (fs, model, cutoff), plus cached scores."""
    scored: dict[tuple, dict] = {}
    rows = []
    for fs, model in combos:
        oof, sc, bp = route_grid_scores(fs, model, train, primary.X, select_k)
        scored[(fs, model)] = {"oof": oof, "scores": sc, "best_params": bp}
        cidx = concordance(sc, primary.time, primary.event)
        for cutoff in cutoffs:
            groups, _ = CUTOFF_METHODS[cutoff](oof, train.rfs_2year, sc)
            n_hi, n_lo = int((groups == "high").sum()), int((groups == "low").sum())
            bal = is_balanced(n_hi, n_lo)
            row = {"fs": fs, "model": model, "cutoff": cutoff, "c_index": cidx,
                   "n_high": n_hi, "n_low": n_lo, "balanced": bal,
                   "hr": None, "logrank_p": None}
            if bal:
                s = analyze_groups(groups, primary.time, primary.event)
                row["hr"] = s["hr_high_vs_low"]
                row["logrank_p"] = s["logrank_p"]
            rows.append(row)
        print(f"  scored {model}/{fs} (params={bp})")
    df = pd.DataFrame(rows)
    cand = df[df["balanced"] & df["logrank_p"].notna()]
    correct = cand[cand["hr"] > 1.0]
    pool = correct if len(correct) else cand
    best = (pool.sort_values("logrank_p").iloc[0] if len(pool) else df.iloc[0])
    return best, scored


def _restricted_rows(groups, scores, cohort_data, cohort, taus):
    """One row per τ (finite horizons + full follow-up) for a fixed grouping."""
    rows = []
    for tau in [*taus, float("inf")]:
        st = restricted_stats(groups, scores, cohort_data.time, cohort_data.event, tau)
        rm = st.get("rmst") or {}
        rows.append({
            "cohort": cohort,
            "tau": ("full" if tau == float("inf") else int(tau)),
            "n_high": st["n_high"], "n_low": st["n_low"],
            "events_high": st["events_high"], "events_low": st["events_low"],
            "logrank_p": st["logrank_p"],
            "hr": st["hr_high_vs_low"], "hr_lo": st["hr_ci_low"], "hr_hi": st["hr_ci_high"],
            "c_index": st["c_index"],
            "rmst_high": rm.get("rmst_high"), "rmst_low": rm.get("rmst_low"),
            "delta_rmst": rm.get("delta_rmst"),
            "delta_ci_low": rm.get("delta_ci_low"), "delta_ci_high": rm.get("delta_ci_high"),
            "rmst_point_p": rm.get("point_p"),
        })
    return rows


def _draw_km(cohort_data, groups, cohort, head, cutoff, taus, out_path):
    """Full-follow-up KM with the τ horizons marked by vertical dashed lines."""
    # ``groups`` may cover only a subset of the cohort (e.g. resection's out-of-fold
    # labelled patients); align time/event to it before plotting/scoring.
    time = cohort_data.time.loc[groups.index]
    event = cohort_data.event.loc[groups.index]
    stats = analyze_groups(groups, time, event)
    stats["c_index"] = concordance(_pseudo(groups), time, event)
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    _draw_subplot(ax, time, event, groups, stats,
                  title=f"{head} · {cohort}", ci_show=True)
    for tau in taus:
        ax.axvline(tau, color="0.5", ls="--", lw=0.8)
        ax.text(tau, 1.0, f"{int(tau)}mo", fontsize=7, color="0.4",
                ha="center", va="bottom")
    fig.suptitle(f"Restricted-time KM — {head} + {cutoff}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved KM → {out_path} (+ .svg)")


def _pseudo(groups):
    return (groups == "high").astype(int)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-id", default="9109a6c2")
    p.add_argument("--ensemble", action="store_true",
                   help="score a mean-probability ensemble over --model-ids (forced head via "
                        "--fs/--model) instead of a single embedding.")
    p.add_argument("--model-ids", nargs="+",
                   help="ensemble: embeddings to average (requires --ensemble, --fs, --model).")
    p.add_argument("--select-k", type=int, default=SELECT_K_DEFAULT)
    p.add_argument("--top", type=int, default=5)
    p.add_argument("--primary", default="soramic")
    p.add_argument("--confirm", default="lusanne")
    p.add_argument("--transfer-csv", type=Path,
                   default=Path("results/eval/grid/grid_transfer_soramic.csv"))
    p.add_argument("--time-col", default="RFS_central")
    p.add_argument("--event-col", default="RFS_central_event")
    p.add_argument("--cutoffs", nargs="+", default=DEFAULT_CUTOFFS)
    p.add_argument("--taus", nargs="+", type=float, default=DEFAULT_TAUS)
    p.add_argument("--output-dir", type=Path, default=Path("results/eval/survival"))
    p.add_argument("--fig-dir", type=Path, default=Path("reports/0706"))
    p.add_argument("--km", action="store_true", help="draw KM curves with τ horizons")
    p.add_argument("--no-resection", dest="include_resection", action="store_false",
                   help="skip the in-sample resection training-cohort table")
    p.add_argument("--freeze-on", choices=["oof", "insample"], default="oof",
                   help="resection scores used to freeze the cutoff threshold: out-of-fold "
                        "(default) or the in-sample refit predictions")
    p.add_argument("--fs", help="force this feature-selection head instead of auto-selecting")
    p.add_argument("--model", help="force this classifier head instead of auto-selecting")
    p.add_argument("--force-cutoff", help="force this cutoff instead of the auto-selected one")
    p.add_argument("--tag", help="suffix for output CSV / KM filenames (avoids clobbering)")
    return p.parse_args()


def run_ensemble(args):
    """Restricted-time analysis for a forced-head mean-probability ensemble.

    Mirrors the forced-head single-embedding path, but scores an :class:`EnsembleGrid`
    over ``--model-ids`` via :func:`route_grid_scores_ensemble`. Requires ``--fs`` and
    ``--model`` (the head chosen by the ensemble grid CV).
    """
    from hcc_multimodal.eval.ensemble import load_ensemble_aligned
    from hcc_multimodal.survival.grid_scores import route_grid_scores_ensemble

    if not (args.model_ids and args.fs and args.model):
        raise SystemExit("--ensemble requires --model-ids, --fs and --model")
    endpoint = {"time_col": args.time_col, "event_col": args.event_col}
    train, blocks = load_ensemble_aligned(args.model_ids, "resection", **endpoint)
    primary, _ = load_ensemble_aligned(args.model_ids, args.primary, **endpoint)
    confirm, _ = load_ensemble_aligned(args.model_ids, args.confirm, **endpoint)

    fs_b, model_b = args.fs, args.model
    cutoff = args.force_cutoff or "kmeans_frozen"
    head = f"{model_b}/{fs_b}"
    tag = "+".join(args.model_ids)
    print(f"Ensemble [{tag}] forced head: {head} + {cutoff} (select_k={args.select_k})")

    oof, sc_primary, bp = route_grid_scores_ensemble(
        fs_b, model_b, train, primary.X, blocks, args.select_k)
    _, sc_confirm, _ = route_grid_scores_ensemble(
        fs_b, model_b, train, confirm.X, blocks, args.select_k)
    _, sc_resection, _ = route_grid_scores_ensemble(
        fs_b, model_b, train, train.X, blocks, args.select_k)
    print(f"  best_params={bp}")
    freeze = sc_resection if args.freeze_on == "insample" else oof

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cohorts = [(args.primary, primary, sc_primary), (args.confirm, confirm, sc_confirm)]
    if args.include_resection:
        cohorts.insert(0, ("resection", train, sc_resection))
    for cohort, cd, sc in cohorts:
        groups, _ = CUTOFF_METHODS[cutoff](freeze, train.rfs_2year, sc)
        rows = _restricted_rows(groups, sc, cd, cohort, args.taus)
        df = pd.DataFrame(rows)
        df.insert(0, "cutoff", cutoff)
        df.insert(0, "head", head)
        suffix = f"_{args.tag}" if args.tag else ""
        path = args.output_dir / f"restricted_time_{cohort}{suffix}.csv"
        df.to_csv(path, index=False)
        print(f"Wrote {path}")
        if args.km:
            _draw_km(cd, groups, cohort, head, cutoff, args.taus,
                     args.fig_dir / f"km_restricted_{cohort}{suffix}.png")


def main():
    args = parse_args()
    if args.ensemble:
        run_ensemble(args)
        return
    transfer = pd.read_csv(args.transfer_csv)
    top = transfer.sort_values("auroc", ascending=False).head(args.top)[["fs", "model"]]
    combos = list(top.itertuples(index=False, name=None))
    print(f"Top {args.top} combos by {args.primary} AUROC: "
          + ", ".join(f"{m}/{fs}" for fs, m in combos))

    endpoint = {"time_col": args.time_col, "event_col": args.event_col}
    train = load_source_aligned(args.model_id, "resection", **endpoint)
    primary = load_source_aligned(args.model_id, args.primary, **endpoint)
    confirm = load_source_aligned(args.model_id, args.confirm, **endpoint)

    best, scored = _select_best_head(combos, train, primary, args.select_k, args.cutoffs)
    if args.fs and args.model:
        fs_b, model_b = args.fs, args.model
        if (fs_b, model_b) not in scored:  # forced head outside the top-N pool
            oof, sc, bp = route_grid_scores(fs_b, model_b, train, primary.X, args.select_k)
            scored[(fs_b, model_b)] = {"oof": oof, "scores": sc, "best_params": bp}
        cutoff = args.force_cutoff or best["cutoff"]
        head = f"{model_b}/{fs_b}"
        print(f"\nForced head on {args.primary}: {head} + {cutoff}")
    else:
        fs_b, model_b = best["fs"], best["model"]
        cutoff = args.force_cutoff or best["cutoff"]
        head = f"{model_b}/{fs_b}"
        print(f"\nBest on {args.primary}: {head} + {cutoff} "
              f"(full-follow-up log-rank p={best['logrank_p']:.3f}, HR={best['hr']:.2f})")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    oof = scored[(fs_b, model_b)]["oof"]
    sc_primary = scored[(fs_b, model_b)]["scores"]
    _, sc_confirm, _ = route_grid_scores(fs_b, model_b, train, confirm.X, args.select_k)

    # In-sample refit resection scores over all 60 patients: the same refit head that
    # transfers to Soramic/Lausanne, applied back to resection. Serves as the resection
    # cohort's own scores and, with --freeze-on insample, as the basis for the frozen
    # cutoff threshold (median of these scores) instead of the out-of-fold scores.
    _, sc_resection, _ = route_grid_scores(fs_b, model_b, train, train.X, args.select_k)
    freeze = sc_resection if args.freeze_on == "insample" else oof

    cohorts = [(args.primary, primary, sc_primary), (args.confirm, confirm, sc_confirm)]
    if args.include_resection:
        cohorts.insert(0, ("resection", train, sc_resection))

    for cohort, cd, sc in cohorts:
        groups, _ = CUTOFF_METHODS[cutoff](freeze, train.rfs_2year, sc)
        rows = _restricted_rows(groups, sc, cd, cohort, args.taus)
        df = pd.DataFrame(rows)
        df.insert(0, "cutoff", cutoff)
        df.insert(0, "head", head)
        suffix = f"_{args.tag}" if args.tag else ""
        path = args.output_dir / f"restricted_time_{cohort}{suffix}.csv"
        df.to_csv(path, index=False)
        print(f"Wrote {path}")
        if args.km:
            _draw_km(cd, groups, cohort, head, cutoff, args.taus,
                     args.fig_dir / f"km_restricted_{cohort}{suffix}.png")


if __name__ == "__main__":
    main()
