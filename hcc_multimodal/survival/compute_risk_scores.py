"""Task 1 — compute Route A / Route B risk scores for each model.

For every (model, route) writes a tidy parquet with the out-of-fold resection
scores (split=train) and the frozen-model ablation scores (split=test) for both
cohorts:

    results/eval/survival/scores/{model_id}_{route}.parquet
    columns: cohort, SID, split, risk_score, time, event

Run:  python -m hcc_multimodal.survival.compute_risk_scores
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from hcc_multimodal.eval.data import PROJECT_ROOT

from . import COHORTS, MODELS
from .data import CohortData, load_aligned
from .risk_scores import route_a_scores, route_b_scores

SCORES_DIR = PROJECT_ROOT / "results" / "eval" / "survival" / "scores"


def _rows(scores: pd.Series, cohort: str, split: str, surv: CohortData) -> pd.DataFrame:
    idx = scores.index
    return pd.DataFrame(
        {
            "cohort": cohort,
            "SID": idx,
            "split": split,
            "risk_score": scores.values,
            "time": surv.time.loc[idx].values,
            "event": surv.event.loc[idx].values,
        }
    )


def compute_for_model(model_id: str, head: str, n_components: int) -> dict[str, pd.DataFrame]:
    train = load_aligned(model_id, "resection")
    test = {c: load_aligned(model_id, c) for c in COHORTS}
    test_X = pd.concat([test[c].X for c in COHORTS])

    oof_a, test_a = route_a_scores(head, train, test_X)
    oof_b, test_b = route_b_scores(train, test_X, n_components=n_components)

    out: dict[str, pd.DataFrame] = {}
    for route, oof, test_scores in (("a", oof_a, test_a), ("b", oof_b, test_b)):
        frames = [_rows(oof, "resection", "train", train)]
        for c in COHORTS:
            cohort_scores = test_scores.loc[test[c].X.index]
            frames.append(_rows(cohort_scores, c, "test", test[c]))
        out[route] = pd.concat(frames, ignore_index=True)
    return out


def main(args: argparse.Namespace) -> None:
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    for model_id, head in MODELS.items():
        tables = compute_for_model(model_id, head, args.n_components)
        for route, df in tables.items():
            path = SCORES_DIR / f"{model_id}_{route}.parquet"
            df.to_parquet(path, index=False)
            n_tr = (df.split == "train").sum()
            n_te = (df.split == "test").sum()
            print(f"{model_id} route {route.upper()}: {n_tr} train + {n_te} test rows → {path.name}")


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-components", type=int, default=10, help="PCA components for Route B (default: %(default)s)")
    return p.parse_args()


if __name__ == "__main__":
    main(_parse())
