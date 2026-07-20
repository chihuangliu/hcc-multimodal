"""Per-dimension KS distribution-drift breakdown (report §6), reusing embedding_drift.

Default: for each (model_id, cohort_pair) prints the distribution of the per-dimension
KS D-statistic (min / p25 / median / p75 / p95 / max) and frac_sig — the detail behind
``embedding_drift.ks_drift``'s median/mean summary (§6.1).

--corr: across every model in ``embedding_drift.MODEL_CONFIGS``, correlates the
resection→cohort KS median D with the transfer AUROC (Spearman ρ) — reproduces §6.2,
showing drift predicts Soramic transfer where resection CV does not.

Usage:
  python -m hcc_multimodal.eval.diagnose.ks a6f970d6 dc7e1d10
  python -m hcc_multimodal.eval.diagnose.ks --corr
"""

import argparse
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr

from hcc_multimodal.eval.diagnose.common import (
    COHORTS,
    LAUSANNE_AUROC,
    SORAMIC_AUROC,
    ks_drift,
    load_embeddings,
    perdim_ks_d,
)
from hcc_multimodal.eval.embedding_drift import MODEL_CONFIGS

_PCTS = [0, 25, 50, 75, 95, 100]
_TRANSFER = {"soramic": SORAMIC_AUROC, "lausanne": LAUSANNE_AUROC}


def perdim_table(model_ids: list[str], labeled: bool = False) -> pd.DataFrame:
    rows = []
    for mid in model_ids:
        embs = {c: load_embeddings(mid, c, labeled=labeled) for c in COHORTS}
        for a, b in combinations(COHORTS, 2):
            d = perdim_ks_d(embs[a], embs[b])
            pvals = np.array([ks_2samp(embs[a][:, i], embs[b][:, i]).pvalue for i in range(len(d))])
            q = dict(zip(["min", "p25", "med", "p75", "p95", "max"], np.percentile(d, _PCTS)))
            rows.append({"model_id": mid, "pair": f"{a}·{b}", **q, "frac_sig": float((pvals < 0.05).mean())})
    return pd.DataFrame(rows)


def corr_table(target: str, labeled: bool = False) -> tuple[pd.DataFrame, float, float]:
    """resection→target KS median D vs transfer AUROC across all MODEL_CONFIGS models."""
    auroc = _TRANSFER[target]
    rows = []
    for mid, suffix in MODEL_CONFIGS:
        if mid not in auroc:
            continue
        res = load_embeddings(mid, "resection", suffix, labeled=labeled)
        tgt = load_embeddings(mid, target, suffix, labeled=labeled)
        stats = ks_drift(res, tgt)  # reused from embedding_drift
        rows.append({"model_id": mid, "ks_median_d": stats["median_d"],
                     "frac_sig": stats["frac_sig"], f"{target}_auroc": auroc[mid]})
    df = pd.DataFrame(rows).sort_values("ks_median_d", ascending=False)
    rho_d = spearmanr(df["ks_median_d"], df[f"{target}_auroc"]).correlation
    rho_s = spearmanr(df["frac_sig"], df[f"{target}_auroc"]).correlation
    return df, rho_d, rho_s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_ids", nargs="*", help="model_ids for the per-dimension breakdown")
    ap.add_argument("--corr", action="store_true", help="drift↔transfer correlation across all models")
    ap.add_argument("--target", default="soramic", choices=["soramic", "lausanne"], help="--corr target cohort")
    ap.add_argument("--labeled-only", action="store_true", help="restrict to 2yr-RFS-labeled SIDs")
    args = ap.parse_args()

    if args.model_ids:
        with pd.option_context("display.float_format", lambda v: f"{v:.3f}"):
            print(perdim_table(args.model_ids, args.labeled_only).to_string(index=False))

    if args.corr:
        df, rho_d, rho_s = corr_table(args.target, args.labeled_only)
        print(f"\nresection→{args.target} KS drift vs transfer (all models):")
        with pd.option_context("display.float_format", lambda v: f"{v:.3f}"):
            print(df.to_string(index=False))
        print(f"\nSpearman ρ(KS median D, {args.target} AUROC) = {rho_d:+.3f}")
        print(f"Spearman ρ(frac_sig,   {args.target} AUROC) = {rho_s:+.3f}")


if __name__ == "__main__":
    main()
