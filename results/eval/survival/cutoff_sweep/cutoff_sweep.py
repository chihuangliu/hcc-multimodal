"""Sweep resection-frozen cutoff methods for the dc7e1d10 Ridge/RF Import. head.

Scores once (route_grid_scores, select_k=43), then applies each *frozen* cutoff
(threshold computed on resection only) to every cohort. Reports split balance and
full-follow-up + tau=24 separation on RFS. Threshold must be leakage-free w.r.t. the
external cohorts, so within-cohort methods are excluded."""
import numpy as np
import pandas as pd

from hcc_multimodal.survival.cutoffs import CUTOFF_METHODS
from hcc_multimodal.survival.data import load_source_aligned
from hcc_multimodal.survival.grid_scores import route_grid_scores
from hcc_multimodal.survival.analysis import analyze_groups, concordance
from hcc_multimodal.survival.restricted import restricted_stats

FROZEN = ["median_frozen", "kmeans_frozen", "kmeans_log_frozen", "youden_frozen"]
FS, MODEL, K = "RF Import.", "Ridge", 43

train = load_source_aligned("dc7e1d10", "resection")
soramic = load_source_aligned("dc7e1d10", "soramic")
lusanne = load_source_aligned("dc7e1d10", "lusanne")

# in-sample refit resection scores (freeze basis) + test scores
_, sc_sor, _ = route_grid_scores(FS, MODEL, train, soramic.X, K)
_, sc_lus, _ = route_grid_scores(FS, MODEL, train, lusanne.X, K)
_, sc_res, _ = route_grid_scores(FS, MODEL, train, train.X, K)
freeze = sc_res  # freeze-on insample

cohorts = [("resection", train, sc_res), ("soramic", soramic, sc_sor), ("lusanne", lusanne, sc_lus)]

rows = []
for method in FROZEN:
    for name, cd, sc in cohorts:
        groups, meta = CUTOFF_METHODS[method](freeze, train.rfs_2year, sc)
        n_hi = int((groups == "high").sum()); n_lo = int((groups == "low").sum())
        # full follow-up
        full = analyze_groups(groups, cd.time, cd.event)
        cidx = concordance(sc, cd.time, cd.event)
        # tau=24
        st24 = restricted_stats(groups, sc, cd.time, cd.event, 24.0)
        rows.append({
            "cutoff": method, "cohort": name, "thr": round(meta["threshold"], 4),
            "n_hi": n_hi, "n_lo": n_lo,
            "hr_full": round(full["hr_high_vs_low"], 2) if n_lo and n_hi else None,
            "lr_full": round(full["logrank_p"], 3) if n_lo and n_hi else None,
            "c_idx": round(cidx, 3),
            "lr_24": round(st24["logrank_p"], 3) if st24.get("logrank_p") is not None else None,
            "pt_24": round(st24["rmst"]["point_p"], 3) if st24.get("rmst") and st24["rmst"].get("point_p") is not None else None,
        })

df = pd.DataFrame(rows)
pd.set_option("display.width", 200)
for method in FROZEN:
    print(f"\n=== {method} ===")
    print(df[df.cutoff == method].drop(columns="cutoff").to_string(index=False))
df.to_csv("/Users/amo/.claude/jobs/1fd3222e/tmp/cutoff_sweep.csv", index=False)
