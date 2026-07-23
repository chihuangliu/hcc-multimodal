"""Survival statistics on the ablation cohorts.

Split into a cutoff-free part (Harrell's C-index, which depends only on the
continuous score ordering) and a group-dependent part (KM medians, log-rank, Cox
HR) that takes a precomputed high/low grouping from :mod:`cutoffs`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
from sklearn.metrics import cohen_kappa_score


def concordance(scores: pd.Series, time: pd.Series, event: pd.Series) -> float | None:
    """Harrell's C-index; higher risk score => shorter survival => negate score."""
    if event.sum() == 0:
        return None
    return float(concordance_index(time, -scores, event))


def is_balanced(n_high: int, n_low: int, min_per_group: int = 5) -> bool:
    return min(n_high, n_low) >= min_per_group


def _km_stats(time: pd.Series, event: pd.Series) -> dict:
    kmf = KaplanMeierFitter().fit(time, event)
    return {
        "median_rfs": _finite(kmf.median_survival_time_),
        "rfs_rate_12mo": _surv_at(kmf, 12.0),
        "rfs_rate_24mo": _surv_at(kmf, 24.0),
    }


def _surv_at(kmf: KaplanMeierFitter, t: float) -> float | None:
    try:
        return float(kmf.predict(t))
    except Exception:
        return None


def _finite(x) -> float | None:
    return None if x is None or not np.isfinite(x) else float(x)


def analyze_groups(groups: pd.Series, time: pd.Series, event: pd.Series) -> dict:
    """KM medians, log-rank p, Cox HR for a high/low grouping (test cohort)."""
    idx = groups.index
    t = time.loc[idx].astype(float)
    e = event.loc[idx].astype(int)
    hi = groups == "high"
    lo = groups == "low"

    out = {
        "n": int(len(idx)),
        "n_high": int(hi.sum()),
        "n_low": int(lo.sum()),
        "events_high": int(e[hi].sum()),
        "events_low": int(e[lo].sum()),
        "logrank_p": None,
        "hr_high_vs_low": None,
        "hr_ci_low": None,
        "hr_ci_high": None,
        "cox_p": None,
        "high": _km_stats(t[hi], e[hi]) if hi.sum() else None,
        "low": _km_stats(t[lo], e[lo]) if lo.sum() else None,
    }

    if hi.sum() > 0 and lo.sum() > 0:
        lr = logrank_test(t[hi], t[lo], event_observed_A=e[hi], event_observed_B=e[lo])
        out["logrank_p"] = float(lr.p_value)

    if hi.sum() > 0 and lo.sum() > 0 and e.sum() > 0:
        cox_df = pd.DataFrame({"time": t.values, "event": e.values, "group": hi.astype(int).values})
        try:
            cph = CoxPHFitter().fit(cox_df, duration_col="time", event_col="event")
            row = cph.summary.loc["group"]
            out["hr_high_vs_low"] = float(row["exp(coef)"])
            out["hr_ci_low"] = float(row["exp(coef) lower 95%"])
            out["hr_ci_high"] = float(row["exp(coef) upper 95%"])
            out["cox_p"] = float(row["p"])
        except Exception:
            pass
    return out


def route_agreement(groups_a: pd.Series, groups_b: pd.Series) -> dict:
    """Cohen's kappa + overlap between two group assignments (same cohort)."""
    idx = groups_a.index.intersection(groups_b.index)
    a, b = groups_a.loc[idx], groups_b.loc[idx]
    overlap = float((a.values == b.values).mean()) if len(idx) else None
    kappa = None
    if len(idx) and a.nunique() > 1 and b.nunique() > 1:
        kappa = float(cohen_kappa_score(a, b))
    return {"n": int(len(idx)), "overlap": overlap, "cohen_kappa": kappa}
