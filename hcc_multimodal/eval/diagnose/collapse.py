"""Representational-collapse metrics for a model's cohort embeddings (report §3).

For each (model_id, cohort) reports:
  n              – number of patients
  norm_mean/std  – L2 norm of the embedding vectors; near-constant norm ⇒ collapse
  norm_cv        – norm std / mean (a6f970d6 ~1% vs dc7e1d10 ~19% on resection)
  cos_mean       – mean pairwise cosine similarity; →1 ⇒ near-collinear vectors
  cos_angle_deg  – arccos(cos_mean), the mean pairwise angle
  eff_rank       – participation ratio of covariance eigenvalues (of dim)

Usage:
  python -m hcc_multimodal.eval.diagnose.collapse a6f970d6 dc7e1d10
  python -m hcc_multimodal.eval.diagnose.collapse a6f970d6 --cohort resection soramic
"""

import argparse

import numpy as np
import pandas as pd

from hcc_multimodal.eval.diagnose.common import COHORTS, load_embeddings


def mean_pairwise_cosine(x: np.ndarray) -> float:
    xn = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
    sim = xn @ xn.T
    iu = np.triu_indices(len(x), k=1)
    return float(sim[iu].mean())


def effective_rank(x: np.ndarray) -> float:
    """Participation ratio of the covariance spectrum: (Σλ)² / Σλ²."""
    xc = x - x.mean(0)
    ev = np.linalg.svd(xc, compute_uv=False) ** 2
    ev = ev / ev.sum()
    return float(1.0 / np.sum(ev**2))


def collapse_metrics(x: np.ndarray) -> dict[str, float]:
    norms = np.linalg.norm(x, axis=1)
    cos = mean_pairwise_cosine(x)
    return {
        "n": len(x),
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "norm_cv": float(norms.std() / (norms.mean() + 1e-9)),
        "cos_mean": cos,
        "cos_angle_deg": float(np.degrees(np.arccos(np.clip(cos, -1, 1)))),
        "eff_rank": effective_rank(x),
    }


def run(model_ids: list[str], cohorts: list[str], labeled: bool = False) -> pd.DataFrame:
    rows = []
    for mid in model_ids:
        for cohort in cohorts:
            x = load_embeddings(mid, cohort, labeled=labeled)
            rows.append({"model_id": mid, "cohort": cohort, **collapse_metrics(x)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_ids", nargs="+", help="one or more model_ids")
    ap.add_argument("--cohort", nargs="+", default=list(COHORTS), choices=COHORTS)
    ap.add_argument("--labeled-only", action="store_true", help="restrict to 2yr-RFS-labeled SIDs")
    ap.add_argument("--out", help="optional CSV path")
    args = ap.parse_args()

    df = run(args.model_ids, args.cohort, args.labeled_only)
    with pd.option_context("display.float_format", lambda v: f"{v:.3f}"):
        print(df.to_string(index=False))
    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
