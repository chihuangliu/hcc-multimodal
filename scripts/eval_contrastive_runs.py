"""Extract → grid → survival evaluation for a set of contrastive runs.

For each model reference (a bare run id = ``best_model.pt``, or ``<run>@<epoch>``):

1. **extract** patient-level image embeddings for resection + each ablation cohort
   (``eval.eval --mode embedding``), cached under the run's ``cached_embeddings/``;
2. **cv-rank** (once, across all refs) — image-only nested-CV resection AUC with the
   best of LR/RF, the "Resection CV AUC" column of the 0803-style report tables;
3. **grid** — the 10 classifier × 13 feature-selection flat 3-fold grid on resection,
   transferred to each cohort, with heatmaps;
4. **survival** — the grid's best-by-CV cell is carried into the restricted-time
   analysis as a forced head, cutoff picked by primary-cohort power.

Finally writes a summary CSV + a markdown table with the report columns:
run, resection CV AUC, per-cohort transfer AUROC at the best cell, the head, and
the τ=24 log-rank / point-in-time p-values.

Usage:
    python scripts/eval_contrastive_runs.py --run-ids 78456720 41c6db8a ... \\
      --tag-prefix lamgrid --results-subdir grid_flat3_lamgrid \\
      --fig-dir reports/0803/flat3_lamgrid --summary-out results/eval/lamgrid_summary.csv

Stages can be skipped (``--skip extract grid``) to re-run only part of the pipeline.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

# Cell identity -> short token used in survival tags, e.g. "Elastic Net" -> "elasticnet".
_TOKEN_STRIP = str.maketrans({" ": "", ".": "", "-": "", "(": "", ")": "", "/": "_"})


def _token(name: str) -> str:
    return name.translate(_TOKEN_STRIP).lower()


def _run(cmd: list[str], *, log: Path | None = None) -> None:
    """Run a subprocess, streaming to `log` if given. Raises on failure."""
    print(f"    $ {' '.join(cmd)}", flush=True)
    if log is None:
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
        return
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as fh:
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, stdout=fh, stderr=subprocess.STDOUT)


def stage_extract(ref: str, cohorts: list[str], log_dir: Path) -> None:
    for cohort in cohorts:
        _run(
            [PYTHON, "-m", "hcc_multimodal.eval.eval", "--mode", "embedding",
             "--model-id", ref, "--ablation-set", cohort, "--target", "rfs_2year"],
            log=log_dir / f"extract_{_token(ref)}_{cohort}.log",
        )


def stage_cv_rank(refs: list[str], out_dir: Path, log_dir: Path) -> None:
    _run(
        [PYTHON, "-m", "hcc_multimodal.eval.embedding_grid_eval", "--task", "cv-rank",
         "--model-ids", *refs, "--classifiers", "LR", "RF", "--outer-folds", "3",
         "--output-dir", str(out_dir)],
        log=log_dir / "cv_rank.log",
    )


def stage_grid(ref: str, out_dir: Path, fig_dir: Path, cohorts: list[str], log_dir: Path) -> None:
    _run(
        [PYTHON, "-m", "hcc_multimodal.eval.embedding_grid_eval", "--task", "grid",
         "--model-id", ref, "--cv-mode", "flat", "--outer-folds", "3", "--cv-repeats", "1",
         "--select-k-fracs", "0.333", "0.667", "1.0", "--cohorts", *cohorts,
         "--output-dir", str(out_dir), "--fig-dir", str(fig_dir)],
        log=log_dir / f"grid_{_token(ref)}.log",
    )


def best_cell(grid_dir: Path, cohorts: list[str]) -> dict:
    """The best-by-CV grid cell, with its transfer AUROCs and tuned select_k.

    Ties on ``cv_auc_mean`` resolve to the first row, matching the grid runner's
    own ``idxmax`` so the reported head agrees with ``grid_best_by_cv.csv``.
    """
    cv = pd.read_csv(grid_dir / "grid_cv_auc.csv")
    top = cv.loc[cv["cv_auc_mean"].idxmax()]
    tied = cv[cv["cv_auc_mean"] == top["cv_auc_mean"]]
    cell = {
        "model": top["model"],
        "fs": top["fs"],
        "grid_cv_auc": float(top["cv_auc_mean"]),
        "n_tied_cells": int(len(tied)),
    }
    for cohort in cohorts:
        tr = pd.read_csv(grid_dir / f"grid_transfer_{cohort}.csv")
        row = tr[(tr["model"] == top["model"]) & (tr["fs"] == top["fs"])]
        cell[f"{cohort}_auroc"] = float(row["auroc"].iloc[0]) if len(row) else float("nan")
        if "select_k" not in cell and len(row):
            cell["select_k"] = _tuned_k(json.loads(row["best_params"].iloc[0]))
    return cell


# Selectors expose their tuned width under different parameter names; missing from
# all three means the cell selects every feature ("All features").
_K_PARAMS = ("selector__k", "selector__max_features", "selector__n_features_to_select")


def _tuned_k(best_params: dict) -> int | None:
    for key in _K_PARAMS:
        if key in best_params:
            return best_params[key]
    return None


def stage_survival(ref: str, cell: dict, tag: str, out_dir: Path, log_dir: Path,
                   primary: str) -> None:
    cmd = [
        PYTHON, "-m", "hcc_multimodal.survival.run_restricted", "--model-id", ref,
        "--fs", str(cell["fs"]), "--model", str(cell["model"]),
    ]
    if cell.get("select_k") is not None:
        cmd += ["--select-k", str(cell["select_k"])]
    cmd += [
        "--freeze-on", "insample", "--select-cutoff-by-power",
        "--cutoffs", "median_frozen", "kmeans_frozen", "youden_frozen",
        "--taus", "12", "24", "36", "48", "--no-resection",
        "--primary", primary, "--output-dir", str(out_dir), "--tag", tag,
    ]
    _run(cmd, log=log_dir / f"survival_{tag}.log")


def read_tau24(out_dir: Path, tag: str, primary: str) -> dict:
    """τ=24 log-rank and RMST point-in-time p from the primary-cohort restricted table."""
    path = out_dir / f"restricted_time_{primary}_{tag}.csv"
    if not path.exists():
        return {"t24_logrank_p": float("nan"), "t24_point_p": float("nan"), "cutoff": ""}
    df = pd.read_csv(path)
    row = df[df["tau"].astype(str) == "24"]
    if not len(row):
        return {"t24_logrank_p": float("nan"), "t24_point_p": float("nan"), "cutoff": ""}
    row = row.iloc[0]
    return {
        "t24_logrank_p": float(row["logrank_p"]),
        "t24_point_p": float(row["rmst_point_p"]),
        "cutoff": row["cutoff"],
        "n_high": row["n_high"],
        "n_low": row["n_low"],
    }


def fmt_head(cell: dict) -> str:
    k = cell.get("select_k")
    return f"{cell['model']} / {cell['fs']} k={k if k is not None else 'all'}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-ids", nargs="+", required=True,
                   help="Model references: bare run id (best_model.pt) or <run>@<epoch>.")
    p.add_argument("--tag-prefix", default="eval",
                   help="Prefix for survival result tags, e.g. 'lamgrid' -> lamgrid_<run>_<head>_rfs.")
    p.add_argument("--results-subdir", default="grid_flat3_eval",
                   help="Grid outputs go to results/eval/<results-subdir>/<run>/.")
    p.add_argument("--cv-rank-dir", default=None,
                   help="Default: results/eval/cv_rank_<tag-prefix>/")
    p.add_argument("--survival-dir", default="results/eval/survival")
    p.add_argument("--fig-dir", default="reports/0803/flat3_eval")
    p.add_argument("--log-dir", default=None, help="Default: logs/eval_<tag-prefix>/")
    p.add_argument("--cohorts", nargs="+", default=["soramic", "lusanne"])
    p.add_argument("--primary", default="soramic",
                   help="Cohort the survival cutoff is powered on and τ=24 read from.")
    p.add_argument("--summary-out", default=None,
                   help="Default: results/eval/<tag-prefix>_summary.csv")
    p.add_argument("--skip", nargs="*", default=[],
                   choices=["extract", "cv-rank", "grid", "survival"],
                   help="Stages to skip (artifacts assumed present).")
    args = p.parse_args()

    tag_prefix = args.tag_prefix
    grid_root = PROJECT_ROOT / "results" / "eval" / args.results_subdir
    cv_rank_dir = Path(args.cv_rank_dir) if args.cv_rank_dir else \
        PROJECT_ROOT / "results" / "eval" / f"cv_rank_{tag_prefix}"
    survival_dir = PROJECT_ROOT / args.survival_dir
    fig_root = PROJECT_ROOT / args.fig_dir
    log_dir = Path(args.log_dir) if args.log_dir else PROJECT_ROOT / "logs" / f"eval_{tag_prefix}"
    summary_out = Path(args.summary_out) if args.summary_out else \
        PROJECT_ROOT / "results" / "eval" / f"{tag_prefix}_summary.csv"
    log_dir.mkdir(parents=True, exist_ok=True)

    if "extract" not in args.skip:
        for ref in args.run_ids:
            print(f"[extract] {ref}", flush=True)
            stage_extract(ref, args.cohorts, log_dir)

    if "cv-rank" not in args.skip:
        print("[cv-rank] all runs", flush=True)
        stage_cv_rank(args.run_ids, cv_rank_dir, log_dir)

    cv_rank = {}
    cv_rank_csv = cv_rank_dir / "cv_rank_image_only.csv"
    if cv_rank_csv.exists():
        cv_rank = pd.read_csv(cv_rank_csv).set_index("model_id").to_dict("index")

    rows = []
    for ref in args.run_ids:
        tok = _token(ref)
        grid_dir = grid_root / tok
        if "grid" not in args.skip:
            print(f"[grid] {ref}", flush=True)
            stage_grid(ref, grid_dir, fig_root / tok, args.cohorts, log_dir)

        cell = best_cell(grid_dir, args.cohorts)
        tag = f"{tag_prefix}_{tok}_{_token(str(cell['model']))}_{_token(str(cell['fs']))}" \
              f"_k{cell.get('select_k', 'all')}_rfs"
        if "survival" not in args.skip:
            print(f"[survival] {ref} — head {fmt_head(cell)}", flush=True)
            stage_survival(ref, cell, tag, survival_dir, log_dir, args.primary)

        row = {"run": ref, "head": fmt_head(cell), "tag": tag, **cell,
               "resection_cv_auc": cv_rank.get(ref, {}).get("cv_auc_mean", float("nan")),
               "cv_rank_head": cv_rank.get(ref, {}).get("best_head", ""),
               **read_tau24(survival_dir, tag, args.primary)}
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_out, index=False)
    print(f"\nWrote {summary_out}")

    cohort_cols = " | ".join(f"{c.capitalize()} heatmap AUC" for c in args.cohorts)
    print(f"\n| Run | Resection CV AUC | {cohort_cols} | Heatmap Head | "
          f"τ=24 log-rank p | τ=24 point-p |")
    print("|---|--:|" + "--:|" * len(args.cohorts) + "---|--:|--:|")
    for r in rows:
        cohorts_md = " | ".join(f"{r.get(f'{c}_auroc', float('nan')):.3f}" for c in args.cohorts)
        print(f"| `{r['run']}` | {r['resection_cv_auc']:.3f} | {cohorts_md} | {r['head']} | "
              f"{r['t24_logrank_p']:.3f} | {r['t24_point_p']:.3f} |")


if __name__ == "__main__":
    main()
