"""Restricted-time survival analysis at fixed horizons τ (12/24/48 mo …).

Two complementary readouts on the *same* high/low grouping, both computed after
capping follow-up at a horizon τ:

  A. Administrative-censoring stratification — every event after τ is censored and
     every follow-up time is clipped to τ, then the usual KM medians / log-rank /
     Cox HR / C-index (:mod:`analysis`) are recomputed. Answers "does the score
     separate recurrence *within* τ months".

  B. Restricted Mean Survival Time (RMST) — the area under each arm's KM curve on
     [0, τ], i.e. the expected event-free months accrued by τ. We report per-arm
     RMST, the high−low difference ΔRMST (95% CI from the two lifelines variances,
     independent groups), and the point-in-time survival-difference test p-value.

Both share :func:`administrative_censor`, so A and B always describe the identical
truncated dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import survival_difference_at_fixed_point_in_time_test
from lifelines.utils import restricted_mean_survival_time

from .analysis import analyze_groups, concordance


def administrative_censor(
    time: pd.Series, event: pd.Series, tau: float
) -> tuple[pd.Series, pd.Series]:
    """Cap follow-up at ``tau``: events after τ become censored, times clip to τ.

    Returns new (time, event) aligned to the inputs. A patient with time > τ is
    event-free-at-τ regardless of their eventual outcome; a patient with time ≤ τ
    keeps their observed time and event.
    """
    t = time.astype(float)
    e = event.astype(int)
    beyond = t > tau
    t_new = t.where(~beyond, float(tau))
    e_new = e.where(~beyond, 0)
    return t_new, e_new


def rmst_arm(time: pd.Series, event: pd.Series, tau: float) -> dict:
    """Per-arm RMST on [0, τ] with its lifelines variance (0 if <1 patient)."""
    if len(time) == 0:
        return {"rmst": None, "var": None, "n": 0}
    kmf = KaplanMeierFitter().fit(time.astype(float), event.astype(int))
    rmst, var = restricted_mean_survival_time(kmf, t=float(tau), return_variance=True)
    return {"rmst": float(rmst), "var": float(var), "n": int(len(time))}


def rmst_difference(
    groups: pd.Series, time: pd.Series, event: pd.Series, tau: float
) -> dict:
    """ΔRMST (high − low) on [0, τ] with a 95% CI and point-in-time test p-value.

    The CI uses Var(ΔRMST) = Var_high + Var_low (arms are independent). ``point_p``
    is the survival-difference-at-τ test (Klein et al.), i.e. S_high(τ) vs S_low(τ);
    it is a companion to the log-rank on the truncated data, not identical to it.
    """
    idx = groups.index
    t = time.loc[idx].astype(float)
    e = event.loc[idx].astype(int)
    hi, lo = groups == "high", groups == "low"

    out = {
        "tau": float(tau),
        "rmst_high": None, "rmst_low": None,
        "delta_rmst": None, "delta_ci_low": None, "delta_ci_high": None,
        "point_p": None,
    }
    if hi.sum() == 0 or lo.sum() == 0:
        return out

    a = rmst_arm(t[hi], e[hi], tau)
    b = rmst_arm(t[lo], e[lo], tau)
    out["rmst_high"] = a["rmst"]
    out["rmst_low"] = b["rmst"]
    delta = a["rmst"] - b["rmst"]
    se = float(np.sqrt(a["var"] + b["var"]))
    out["delta_rmst"] = float(delta)
    out["delta_ci_low"] = float(delta - 1.96 * se)
    out["delta_ci_high"] = float(delta + 1.96 * se)

    if e[hi].sum() > 0 or e[lo].sum() > 0:
        try:
            kmf_hi = KaplanMeierFitter().fit(t[hi], e[hi], label="high")
            kmf_lo = KaplanMeierFitter().fit(t[lo], e[lo], label="low")
            res = survival_difference_at_fixed_point_in_time_test(
                float(tau), kmf_hi, kmf_lo
            )
            out["point_p"] = float(res.p_value)
        except Exception:
            pass
    return out


def restricted_stats(
    groups: pd.Series, scores: pd.Series, time: pd.Series, event: pd.Series, tau: float
) -> dict:
    """A (censored KM/log-rank/HR/C-index) + B (RMST) for one grouping at τ.

    ``scores`` is the continuous risk score (for Harrell's C-index, matching the
    full-follow-up convention). ``tau=inf`` reproduces the full-follow-up analysis
    (no censoring), so it is a valid reference row.
    """
    idx = groups.index
    t = time.loc[idx].astype(float)
    e = event.loc[idx].astype(int)

    if np.isfinite(tau):
        t_c, e_c = administrative_censor(t, e, tau)
    else:
        t_c, e_c = t, e

    stats = analyze_groups(groups, t_c, e_c)  # A: on the truncated data
    stats["c_index"] = concordance(scores.loc[idx], t_c, e_c)
    stats["tau"] = float(tau)
    stats["rmst"] = rmst_difference(groups, t_c, e_c, tau) if np.isfinite(tau) else None
    return stats
