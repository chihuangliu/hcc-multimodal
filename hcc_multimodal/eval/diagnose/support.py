"""Out-of-support and centroid-shift diagnostics from a src (train) to a dst cohort (report §4).

For each (model_id, src, dst) reports:
  oos_frac       – fraction of (point, dim) cells of dst outside src's per-dim [min, max]
                   (axis-aligned bounding box; a loose over-approximation of the support,
                    so true extrapolation is at least this high)
  centroid_gap   – ||(μ_src − μ_dst) / σ_src||₂ over dims, in src-σ units. This is the
                   shift the classifier sees, since StandardScaler divides by σ_src.
  centroid_per_dim – centroid_gap / √dim, the avg σ-shift per dimension
  proxy_a_auc    – 5-fold CV AUROC of a logistic domain classifier src-vs-dst. Saturates
                   at ~1.0 for small n / high dim, so it is reported but NOT discriminative.

Usage:
  python -m hcc_multimodal.eval.diagnose.support a6f970d6 --src resection --dst soramic
  python -m hcc_multimodal.eval.diagnose.support a6f970d6 dc7e1d10 --src resection --dst soramic lausanne
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from hcc_multimodal.eval.diagnose.common import COHORTS, load_embeddings


def oos_fraction(src: np.ndarray, dst: np.ndarray) -> float:
    lo, hi = src.min(0), src.max(0)
    return float(((dst < lo) | (dst > hi)).mean())


def centroid_gap(src: np.ndarray, dst: np.ndarray) -> float:
    sd = src.std(0) + 1e-9
    return float(np.linalg.norm((src.mean(0) - dst.mean(0)) / sd))


def proxy_a_auc(src: np.ndarray, dst: np.ndarray, seed: int = 0) -> float:
    x = np.vstack([src, dst])
    y = np.r_[np.zeros(len(src)), np.ones(len(dst))]
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    return float(cross_val_score(clf, x, y, cv=cv, scoring="roc_auc").mean())


def run(model_ids: list[str], src: str, dsts: list[str], seed: int = 0, labeled: bool = False) -> pd.DataFrame:
    rows = []
    for mid in model_ids:
        a = load_embeddings(mid, src, labeled=labeled)
        for dst in dsts:
            b = load_embeddings(mid, dst, labeled=labeled)
            gap = centroid_gap(a, b)
            rows.append({
                "model_id": mid, "src": src, "dst": dst,
                "oos_frac": oos_fraction(a, b),
                "centroid_gap": gap,
                "centroid_per_dim": gap / np.sqrt(a.shape[1]),
                "proxy_a_auc": proxy_a_auc(a, b, seed),
            })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_ids", nargs="+", help="one or more model_ids")
    ap.add_argument("--src", default="resection", choices=COHORTS, help="training cohort (support)")
    ap.add_argument("--dst", nargs="+", default=["soramic", "lausanne"], choices=COHORTS, help="target cohort(s)")
    ap.add_argument("--seed", type=int, default=0, help="proxy-A CV seed")
    ap.add_argument("--labeled-only", action="store_true", help="restrict to 2yr-RFS-labeled SIDs")
    ap.add_argument("--out", help="optional CSV path")
    args = ap.parse_args()

    df = run(args.model_ids, args.src, args.dst, args.seed, args.labeled_only)
    with pd.option_context("display.float_format", lambda v: f"{v:.3f}"):
        print(df.to_string(index=False))
    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
