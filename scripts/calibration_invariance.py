"""Test whether Platt calibration can improve AUROC / C-index / log-rank p.

Motivation
----------
``hcc_multimodal.eval.calibration`` re-aligns a head's probability outputs to a
test cohort's positive rate. AUROC, Harrell's C-index and any *data-driven-cutoff*
log-rank p are rank-based, so a monotonic recalibration cannot change them. This
script confirms that empirically and, crucially, contrasts two calibrators:

* ``global``  - a single Platt sigmoid fit on the whole labelled test set. Strictly
  monotonic, so rank-based metrics are invariant (up to score-saturation ties).
* ``cv``      - the 5-fold cross-validated Platt used by ``calibration._platt_cv``.
  A different sigmoid per fold makes the pooled transform only piecewise-monotonic,
  which scrambles ranks on small cohorts and *degrades* the metrics.

For each top-N (feature-selection, classifier) head by ``--transfer-csv`` AUROC it
reports AUROC / C-index / median-within log-rank p under raw, cv-Platt and
global-Platt scores, on the labelled subset (rfs_2year notna, which also has
time+event) of each cohort.

Example
-------
    python scripts/calibration_invariance.py \\
        --model-id 9109a6c2 --top 5 --cohorts soramic lusanne
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from hcc_multimodal.eval.calibration import _platt_cv
from hcc_multimodal.eval.grid import SELECT_K_DEFAULT
from hcc_multimodal.survival.analysis import analyze_groups, concordance
from hcc_multimodal.survival.cutoffs import median_within
from hcc_multimodal.survival.data import CohortData, load_source_aligned
from hcc_multimodal.survival.grid_scores import route_grid_scores


def _platt_global(y, raw, n_folds: int = 0, seed: int = 0):
    """Single Platt sigmoid fit on the whole labelled test set (monotonic)."""
    lr = LogisticRegression(solver="lbfgs", max_iter=1000)
    lr.fit(raw.reshape(-1, 1), y)
    return lr.predict_proba(raw.reshape(-1, 1))[:, 1]


CALIBRATORS = {"cv": _platt_cv, "global": _platt_global}


def _labelled_index(scores: pd.Series, cd: CohortData) -> pd.Index:
    mask = cd.rfs_2year.notna()
    return scores.index.intersection(cd.rfs_2year[mask].index).sort_values()


def _metrics(scores: pd.Series, cd: CohortData, idx: pd.Index) -> dict:
    s = scores.loc[idx]
    y = cd.rfs_2year.loc[idx].astype(int)
    t, e = cd.time.loc[idx], cd.event.loc[idx]
    auroc = float(roc_auc_score(y, s)) if y.nunique() == 2 else float("nan")
    groups, _ = median_within(None, None, s)  # test-median split (rank-invariant)
    st = analyze_groups(groups, t, e)
    return {
        "auroc": auroc,
        "c_index": concordance(s, t, e),
        "logrank_p": st.get("logrank_p"),
        "hr": st.get("hr_high_vs_low"),
        "n_high": int((groups == "high").sum()),
        "n_low": int((groups == "low").sum()),
    }


def run(args: argparse.Namespace) -> pd.DataFrame:
    top = (
        pd.read_csv(args.transfer_csv)
        .sort_values("auroc", ascending=False)
        .head(args.top)[["fs", "model"]]
    )
    combos = list(top.itertuples(index=False, name=None))
    print(f"Top {args.top} heads: " + ", ".join(f"{m}/{fs}" for fs, m in combos))

    train = load_source_aligned(args.model_id, args.train_cohort)
    rows = []
    for cohort in args.cohorts:
        cd = load_source_aligned(args.model_id, cohort)
        for fs, model in combos:
            _, test_scores, _ = route_grid_scores(fs, model, train, cd.X, args.select_k)
            idx = _labelled_index(test_scores, cd)
            y = cd.rfs_2year.loc[idx].astype(int).to_numpy()
            raw = test_scores.loc[idx]

            row = {"cohort": cohort, "head": f"{model}/{fs}", "n": len(idx)}
            m_raw = _metrics(raw, cd, idx)
            row.update({f"{k}_raw": v for k, v in m_raw.items()})
            for name, fn in CALIBRATORS.items():
                cal = pd.Series(
                    fn(y, raw.to_numpy(), n_folds=args.n_folds, seed=args.seed),
                    index=idx,
                )
                m = _metrics(cal, cd, idx)
                row.update({f"{k}_{name}": v for k, v in m.items()})
            rows.append(row)

    df = pd.DataFrame(rows)
    _report(df, args.cohorts)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"\nSaved -> {args.output}")
    return df


def _report(df: pd.DataFrame, cohorts: list[str]) -> None:
    pd.set_option("display.width", 220, "display.max_columns", 40)
    for cohort in cohorts:
        sub = df[df.cohort == cohort]
        print(f"\n=== {cohort} (median-within split, labelled subset) ===")
        print(sub[[
            "head", "n",
            "auroc_raw", "auroc_cv", "auroc_global",
            "c_index_raw", "c_index_cv", "c_index_global",
            "logrank_p_raw", "logrank_p_cv", "logrank_p_global",
        ]].round(4).to_string(index=False))

    for name in ("cv", "global"):
        da = (df["auroc_raw"] - df[f"auroc_{name}"]).abs().max()
        dc = (df["c_index_raw"] - df[f"c_index_{name}"]).abs().max()
        dp = (df["logrank_p_raw"] - df[f"logrank_p_{name}"]).abs().max()
        tag = "CV-Platt (calibration._platt_cv)" if name == "cv" else "single global Platt"
        print(f"\n-- {tag} vs raw --")
        print(f"max |dAUROC|={da:.2e}  max |dC-index|={dc:.2e}  max |dlog-rank p|={dp:.2e}")


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model-id", default="9109a6c2")
    p.add_argument("--train-cohort", default="resection")
    p.add_argument("--cohorts", nargs="+", default=["soramic", "lusanne"])
    p.add_argument("--top", type=int, default=5)
    p.add_argument("--select-k", type=int, default=SELECT_K_DEFAULT)
    p.add_argument("--n-folds", type=int, default=5, help="CV folds for CV-Platt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--transfer-csv",
        type=Path,
        default=Path("results/eval/grid/grid_transfer_soramic.csv"),
    )
    p.add_argument(
        "--output",
        default="results/eval/calibration/calibration_invariance.csv",
    )
    return p.parse_args()


if __name__ == "__main__":
    run(_parse())
