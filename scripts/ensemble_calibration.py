"""Calibration assessment + recalibration of the frozen top-k model ensemble.

Discrimination is not the question here: AUROC/C-index are rank statistics and any
monotone recalibration leaves them alone (see ``reports/0706/0706_calibration_invariance.md``).
The question a risk-stratification claim raises is whether the ensemble's *probabilities*
mean what they say — so this script reports Brier / ECE / calibration slope + intercept and
reliability curves, both raw and after recalibration.

The recalibrators are fitted on **resection out-of-fold scores only** (never on test labels),
which is the deployable direction; the earlier ``hcc_multimodal/eval/calibration.py`` fitted a
per-fold Platt scaler *on the test cohort*, which both leaks and breaks monotonicity.
``--oracle`` adds an in-cohort refit as an attainable-ceiling reference.

Usage:
    python scripts/ensemble_calibration.py \
      --model-id d7085bf5 \
      --members-csv results/eval/grid_flat3_bestckpt/d7085bf5/model_ensemble_members.csv \
      --cohorts soramic lusanne \
      --out-dir results/eval/calibration/ensemble_d7085bf5 \
      --fig-dir reports/0831/calibration
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold

from hcc_multimodal.eval.ensemble import HeteroEnsembleGrid, build_member
from hcc_multimodal.eval.grid import SELECT_K_DEFAULT, positive_scores
from hcc_multimodal.survival.cutoffs import kmeans_frozen
from hcc_multimodal.survival.data import load_source_aligned
from hcc_multimodal.train.config import RANDOM_STATE

matplotlib.rcParams["svg.fonttype"] = "none"
EPS = 1e-6

# Internal cohort keys -> the names used in write-ups.
DISPLAY = {"soramic": "SORAMIC", "lusanne": "Lausanne", "resection": "Resection"}


def display_name(cohort):
    return DISPLAY.get(cohort, cohort)


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------
def _clip(p):
    return np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)


def ece(y, p, n_bins, strategy="quantile"):
    """Expected calibration error: |mean(p) - mean(y)| averaged over bins, weighted by bin size.

    Quantile bins (equal count) rather than equal width — at n<70 equal-width bins leave
    most bins empty and the statistic is dominated by whichever bin caught 2 patients.
    """
    p = _clip(p)
    y = np.asarray(y, dtype=float)
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(0, 1, n_bins + 1)
    if len(edges) < 2:
        return np.nan
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, len(edges) - 2)
    total = 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        if not m.any():
            continue
        total += m.sum() * abs(p[m].mean() - y[m].mean())
    return float(total / len(p))


def spiegelhalter_z(y, p):
    """Spiegelhalter's Z-test of the null 'these probabilities are correct'.

    A decomposition of the Brier score under the null; returns (z, two-sided p). Unlike
    Hosmer-Lemeshow it needs no binning, which matters at n<70.
    """
    y = np.asarray(y, dtype=float)
    p = _clip(p)
    num = np.sum((y - p) * (1 - 2 * p))
    den = np.sqrt(np.sum((1 - 2 * p) ** 2 * p * (1 - p)))
    if den == 0:
        return np.nan, np.nan
    z = float(num / den)
    return z, float(2 * (1 - norm.cdf(abs(z))))


def calibration_slope_intercept(y, p):
    """(slope, intercept) of the logistic recalibration curve y ~ a + b*logit(p).

    Perfect calibration is slope 1 / intercept 0. Slope < 1 means the scores are
    over-dispersed (too extreme in both directions); intercept != 0 means the whole
    risk scale is shifted against the observed event rate.
    """
    y = np.asarray(y, dtype=int)
    z = logit(_clip(p)).reshape(-1, 1)
    if len(np.unique(y)) < 2 or np.ptp(z) == 0:
        return np.nan, np.nan
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=10000).fit(z, y)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def metrics_row(y, p, n_bins):
    y = np.asarray(y, dtype=int)
    p = _clip(p)
    slope, intercept = calibration_slope_intercept(y, p)
    z, z_p = spiegelhalter_z(y, p)
    return {
        "n": int(len(y)),
        "prevalence": float(y.mean()),
        "mean_pred": float(p.mean()),
        "citl": float(p.mean() - y.mean()),  # calibration-in-the-large
        "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
        "brier": float(brier_score_loss(y, p)),
        "ece": ece(y, p, n_bins),
        "cal_slope": slope,
        "cal_intercept": intercept,
        "spiegelhalter_z": z,
        "spiegelhalter_p": z_p,
    }


def bootstrap_ci(y, p, n_bins, n_boot, seed, stats=("brier", "ece")):
    """Percentile 95% CI for the listed statistics, resampling patients."""
    if n_boot <= 0:
        return {}
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y, dtype=int), _clip(p)
    draws = {s: [] for s in stats}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        yb, pb = y[idx], p[idx]
        if len(np.unique(yb)) < 2:
            continue
        draws["brier"].append(brier_score_loss(yb, pb))
        draws["ece"].append(ece(yb, pb, n_bins))
    out = {}
    for s in stats:
        v = np.asarray(draws[s], dtype=float)
        v = v[~np.isnan(v)]
        if len(v):
            out[f"{s}_lo"], out[f"{s}_hi"] = float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
    return out


# ---------------------------------------------------------------------------
# Recalibrators — all fitted on resection OOF scores, applied unchanged elsewhere
# ---------------------------------------------------------------------------
def fit_platt(p, y):
    z = logit(_clip(p)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=10000).fit(z, np.asarray(y, dtype=int))
    return lambda q: lr.predict_proba(logit(_clip(q)).reshape(-1, 1))[:, 1]


def fit_isotonic(p, y):
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(_clip(p), np.asarray(y, dtype=float))
    return lambda q: iso.predict(_clip(q))


def prior_shift(fn, p_train, p_target):
    """Wrap a calibrator with a log-odds offset for a known target base rate.

    Deployment-realistic: uses the target cohort's *marginal* prevalence only, no
    individual labels. Corrects the part of the miscalibration that is pure case-mix.
    """
    off = logit(p_target) - logit(p_train)
    return lambda q: expit(logit(_clip(fn(q))) + off)


# ---------------------------------------------------------------------------
# Ensemble scoring
# ---------------------------------------------------------------------------
def load_members(members_csv, select_k, memory=None):
    df = pd.read_csv(members_csv)
    members, labels = [], []
    for _, r in df.iterrows():
        params = json.loads(r["best_params"])
        members.append(build_member(r["model"], r["fs"], params, select_k, memory=memory))
        labels.append(f"{r['model']}/{r['fs']}")
    return members, labels


def labelled(cd):
    m = cd.rfs_2year.notna()
    return cd.X[m], cd.rfs_2year[m].astype(int)


def ensemble_scores(model_id, members_csv, cohorts, select_k, n_folds):
    """Resection OOF scores + per-cohort transfer scores for the frozen ensemble."""
    X_tr, y_tr = labelled(load_source_aligned(model_id, "resection"))

    def _est():
        return HeteroEnsembleGrid(load_members(members_csv, select_k)[0])

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    oof = pd.Series(index=X_tr.index, dtype=float)
    for tr, va in skf.split(X_tr, y_tr):
        est = _est().fit(X_tr.iloc[tr], y_tr.iloc[tr])
        oof.iloc[va] = positive_scores(est, X_tr.iloc[va])

    full = _est().fit(X_tr, y_tr)
    tests = {}
    for c in cohorts:
        Xc, yc = labelled(load_source_aligned(model_id, c))
        tests[c] = (pd.Series(positive_scores(full, Xc), index=Xc.index), yc)
    return oof, y_tr, tests


# ---------------------------------------------------------------------------
# Classification performance at a threshold frozen on the training cohort
# ---------------------------------------------------------------------------
def youden_threshold(y, p):
    """Threshold maximising Youden's J = sensitivity + specificity - 1.

    J is rank-based, so it is invariant to a strictly monotone recalibration: selecting on
    the calibrated scores and selecting on the raw scores flag the *same* patients. It is
    computed on the calibrated scale only so the reported number shares the scale of the
    reported probabilities.
    """
    fpr, tpr, thr = roc_curve(np.asarray(y, dtype=int), _clip(p))
    return float(thr[int(np.argmax(tpr - fpr))])


def classification_row(y, p, thr, n_boot, seed):
    """Rank metrics + the operating point at ``thr``, with bootstrap CIs.

    PR-AUC's no-skill baseline is the cohort prevalence, which moves from 48% (resection)
    to 68/74% (ablation cohorts) — reported alongside so the lift is readable, which a bare
    AUROC hides.
    """
    y = np.asarray(y, dtype=int)
    p = _clip(p)
    tn, fp, fn, tp = confusion_matrix(y, (p >= thr).astype(int), labels=[0, 1]).ravel()
    row = {
        "n": len(y), "prevalence": float(y.mean()), "threshold": float(thr),
        "auroc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "no_skill": float(y.mean()),
        "brier": float(brier_score_loss(y, p)),
        "sens": tp / (tp + fn) if tp + fn else np.nan,
        "spec": tn / (tn + fp) if tn + fp else np.nan,
        "ppv": tp / (tp + fp) if tp + fp else np.nan,
        "npv": tn / (tn + fn) if tn + fn else np.nan,
        "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }
    if n_boot > 0:
        rng = np.random.default_rng(seed)
        au, pa = [], []
        for _ in range(n_boot):
            i = rng.integers(0, len(y), len(y))
            if len(np.unique(y[i])) < 2:
                continue
            au.append(roc_auc_score(y[i], p[i]))
            pa.append(average_precision_score(y[i], p[i]))
        row["auroc_lo"], row["auroc_hi"] = np.percentile(au, [2.5, 97.5])
        row["pr_auc_lo"], row["pr_auc_hi"] = np.percentile(pa, [2.5, 97.5])
    return row


def roc_plot(curves, thr, path, title=None):
    """ROC per cohort with the frozen operating point marked.

    One curve per cohort: the ROC is identical calibrated or not (rank-invariant), so a
    single curve serves both. ``title`` is left off by default — the figure is captioned
    where it is used, and a baked-in title duplicates that caption.
    """
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    for (name, (y, p, color)) in curves.items():
        y = np.asarray(y, dtype=int)
        p = _clip(p)
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, "-", color=color, linewidth=2,
                label=f"{display_name(name)} (AUROC {roc_auc_score(y, p):.3f})")
        yhat = p >= thr
        tn, fp, fn, tp = confusion_matrix(y, yhat.astype(int), labels=[0, 1]).ravel()
        ax.plot(fp / (fp + tn), tp / (tp + fn), "o", color=color, markersize=9,
                markeredgecolor="white", markeredgewidth=1.4, zorder=5)
    ax.plot([0, 1], [0, 1], "k:", linewidth=1, label="chance")
    ax.set_xlabel("1 - specificity", fontsize=12)
    ax.set_ylabel("Sensitivity", fontsize=12)
    if title:
        ax.set_title(title, fontsize=12)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=150)
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path.with_suffix('.png')}")


# ---------------------------------------------------------------------------
# Reliability plot
# ---------------------------------------------------------------------------
def reliability_plot(panels, n_bins, path, title):
    """One subplot per cohort; one line per calibrator, quantile-binned."""
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.4), squeeze=False)
    for ax, (name, series) in zip(axes[0], panels.items()):
        ax.plot([0, 1], [0, 1], "k:", linewidth=1, label="perfect")
        for lbl, (y, p, color) in series.items():
            p = _clip(p)
            y = np.asarray(y, dtype=float)
            edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
            xs, ys = [], []
            idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, len(edges) - 2)
            for b in range(len(edges) - 1):
                m = idx == b
                if m.any():
                    xs.append(p[m].mean())
                    ys.append(y[m].mean())
            ax.plot(xs, ys, "o-", color=color, linewidth=1.8, markersize=6, label=lbl)
        ax.axhline(np.mean(list(series.values())[0][0]), color="grey", linestyle="--",
                   linewidth=1, alpha=0.6, label="observed rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("Predicted probability", fontsize=12)
        ax.set_ylabel("Observed 2-year recurrence rate", fontsize=12)
        ax.set_title(name, fontsize=13)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, loc="lower right")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=150)
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path.with_suffix('.png')}")


# ---------------------------------------------------------------------------
# Does recalibration move the deployed KM risk groups?
# ---------------------------------------------------------------------------
def stratification_check(model_id, members_csv, select_k, oof, y_res, cohorts, calibrators,
                         freeze_on="insample"):
    """Re-derive the frozen k-means cutoff on recalibrated scores and compare the split.

    The deployed survival head splits patients at a k-means boundary *fit on the resection
    OOF scores and frozen*, so it lives on the probability scale and is not automatically
    calibration-invariant the way a within-cohort median split would be. Applying the same
    monotone recalibrator to both sides is the honest comparison: the cutoff is re-derived
    on the recalibrated resection scores, so any change in the split is real.

    Scores here cover every patient with survival follow-up (not just the ``rfs_2year``
    subset), matching the KM tables. ``freeze_on`` mirrors ``run_restricted.py``'s flag —
    ``"insample"`` reproduces the deployed A2 head (cutoff 0.463, Soramic 83/17).
    """
    res_cd = load_source_aligned(model_id, "resection")
    X_tr, y_tr = labelled(res_cd)
    full = HeteroEnsembleGrid(load_members(members_csv, select_k)[0]).fit(X_tr, y_tr)
    # In-sample freeze scores every resection patient with follow-up (n=60), not just the
    # labelled subset — matching run_restricted.py's ``sc_resection``.
    freeze = (pd.Series(positive_scores(full, res_cd.X), index=res_cd.X.index)
              if freeze_on == "insample" else oof)

    rows = []
    # Resection is a row of the thesis KM table too (in-sample), so it is checked as well.
    for c in ["resection", *cohorts]:
        cd = res_cd if c == "resection" else load_source_aligned(model_id, c)
        s_test = pd.Series(positive_scores(full, cd.X), index=cd.X.index)
        base_groups, base_meta = kmeans_frozen(freeze, y_res, s_test)
        for lbl, fn in calibrators.items():
            freeze_c = pd.Series(fn(freeze.values), index=freeze.index)
            s_c = pd.Series(fn(s_test.values), index=s_test.index)
            groups, meta = kmeans_frozen(freeze_c, y_res, s_c)
            rows.append({
                "cohort": c, "calibrator": lbl,
                "cutoff_raw": base_meta["threshold"], "cutoff_recal": meta["threshold"],
                "n": len(groups),
                "n_high_raw": int((base_groups == "high").sum()),
                "n_high_recal": int((groups == "high").sum()),
                "n_reassigned": int((groups != base_groups).sum()),
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--members-csv", type=Path, required=True,
                    help="model_ensemble_members.csv from the grid runner")
    ap.add_argument("--cohorts", nargs="+", default=["soramic", "lusanne"])
    ap.add_argument("--select-k", type=int, default=SELECT_K_DEFAULT)
    ap.add_argument("--n-folds", type=int, default=3, help="resection OOF folds")
    ap.add_argument("--n-bins", type=int, default=4, help="reliability / ECE bins")
    ap.add_argument("--n-boot", type=int, default=2000, help="0 disables bootstrap CIs")
    ap.add_argument("--seed", type=int, default=RANDOM_STATE)
    ap.add_argument("--freeze-on", choices=["oof", "insample"], default="insample",
                    help="resection scores the k-means KM cutoff is frozen on; 'insample' "
                         "matches the deployed A2 survival head")
    ap.add_argument("--threshold-rule", choices=["youden", "prevalence", "fixed"],
                    default="youden",
                    help="how the classification threshold is chosen on the resection "
                         "training cohort before being frozen")
    ap.add_argument("--threshold", type=float,
                    help="the threshold value when --threshold-rule fixed")
    ap.add_argument("--oracle", action="store_true",
                    help="also fit a Platt scaler in-cohort (uses test labels; ceiling only)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--fig-dir", type=Path, required=True)
    args = ap.parse_args()

    oof, y_res, tests = ensemble_scores(args.model_id, args.members_csv, args.cohorts,
                                        args.select_k, args.n_folds)
    print(f"resection OOF n={len(oof)} prev={y_res.mean():.3f} "
          f"score range [{oof.min():.3f}, {oof.max():.3f}]")

    platt = fit_platt(oof.values, y_res.values)
    iso = fit_isotonic(oof.values, y_res.values)
    p_train = float(y_res.mean())

    rows, panels = [], {}

    # -- resection, nested: calibrator refitted inside a CV of the OOF scores so the
    #    training-cohort numbers are not read off their own fit.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    nested = {"platt": np.full(len(oof), np.nan), "isotonic": np.full(len(oof), np.nan)}
    for tr, va in skf.split(oof.values.reshape(-1, 1), y_res.values):
        nested["platt"][va] = fit_platt(oof.values[tr], y_res.values[tr])(oof.values[va])
        nested["isotonic"][va] = fit_isotonic(oof.values[tr], y_res.values[tr])(oof.values[va])

    res_variants = {"uncalibrated": oof.values,
                    "platt (nested)": nested["platt"],
                    "isotonic (nested)": nested["isotonic"]}
    for lbl, p in res_variants.items():
        row = {"cohort": "resection (OOF)", "calibrator": lbl}
        row.update(metrics_row(y_res.values, p, args.n_bins))
        row.update(bootstrap_ci(y_res.values, p, args.n_bins, args.n_boot, args.seed))
        rows.append(row)
    panels["Resection (OOF, n=%d)" % len(oof)] = {
        "uncalibrated": (y_res.values, oof.values, "tab:red"),
        "Platt (nested)": (y_res.values, nested["platt"], "tab:blue"),
    }

    # -- external cohorts: calibrators frozen from resection
    for c, (s, y) in tests.items():
        variants = {
            "uncalibrated": s.values,
            "platt (resection-fit)": platt(s.values),
            "isotonic (resection-fit)": iso(s.values),
            "platt + prior shift": prior_shift(platt, p_train, float(y.mean()))(s.values),
        }
        if args.oracle:
            variants["platt (in-cohort oracle)"] = fit_platt(s.values, y.values)(s.values)
        for lbl, p in variants.items():
            row = {"cohort": c, "calibrator": lbl}
            row.update(metrics_row(y.values, p, args.n_bins))
            row.update(bootstrap_ci(y.values, p, args.n_bins, args.n_boot, args.seed))
            rows.append(row)
        panels[f"{c} (n={len(y)})"] = {
            "uncalibrated": (y.values, s.values, "tab:red"),
            "Platt (resection-fit)": (y.values, variants["platt (resection-fit)"], "tab:blue"),
            "Platt + prior shift": (y.values, variants["platt + prior shift"], "tab:green"),
        }

    df = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv = args.out_dir / f"calibration_{args.model_id}.csv"
    df.to_csv(csv, index=False)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nWrote {csv}")

    # per-patient scores, for any follow-up threshold analysis
    long = [pd.DataFrame({"cohort": "resection (OOF)", "sid": oof.index,
                          "y": y_res.values, "p_raw": oof.values,
                          "p_platt": nested["platt"], "p_iso": nested["isotonic"]})]
    for c, (s, y) in tests.items():
        long.append(pd.DataFrame({"cohort": c, "sid": s.index, "y": y.values,
                                  "p_raw": s.values, "p_platt": platt(s.values),
                                  "p_iso": iso(s.values)}))
    pd.concat(long).to_csv(args.out_dir / f"scores_{args.model_id}.csv", index=False)

    strat = stratification_check(args.model_id, args.members_csv, args.select_k, oof, y_res,
                                 args.cohorts, {"platt": platt, "isotonic": iso},
                                 freeze_on=args.freeze_on)
    strat.to_csv(args.out_dir / f"stratification_{args.model_id}.csv", index=False)
    print("\nKM risk-group stability under recalibration (frozen k-means cutoff):")
    print(strat.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # -- classification at a threshold frozen on the resection training cohort ------
    pr_res = platt(oof.values)
    if args.threshold_rule == "youden":
        thr = youden_threshold(y_res.values, pr_res)
    elif args.threshold_rule == "prevalence":
        thr = float(y_res.mean())
    else:
        if args.threshold is None:
            raise SystemExit("--threshold-rule fixed requires --threshold")
        thr = args.threshold
    print(f"\nClassification threshold ({args.threshold_rule}, frozen on resection): {thr:.4f}")

    crows = [{"cohort": "resection (OOF)", "calibrator": "platt (nested)",
              **classification_row(y_res.values, nested["platt"], thr, args.n_boot, args.seed)}]
    curves = {}
    for c, (s_, y_) in tests.items():
        pc = platt(s_.values)
        crows.append({"cohort": c, "calibrator": "platt (resection-fit)",
                      **classification_row(y_.values, pc, thr, args.n_boot, args.seed)})
        curves[c] = (y_.values, pc, "tab:blue" if len(curves) == 0 else "tab:orange")
    cdf = pd.DataFrame(crows)
    cdf.to_csv(args.out_dir / f"classification_{args.model_id}.csv", index=False)
    show = ["cohort", "n", "prevalence", "auroc", "auroc_lo", "auroc_hi", "pr_auc",
            "pr_auc_lo", "pr_auc_hi", "no_skill", "brier"]
    print(cdf[[c for c in show if c in cdf]].to_string(index=False,
                                                       float_format=lambda v: f"{v:.3f}"))
    print(cdf[["cohort", "sens", "spec", "ppv", "npv", "f1", "tp", "fp", "fn", "tn"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    roc_plot(curves, thr, args.fig_dir / f"roc_{args.model_id}")

    reliability_plot(panels, args.n_bins, args.fig_dir / f"reliability_{args.model_id}",
                     f"Ensemble calibration — {args.model_id} "
                     f"({', '.join(load_members(args.members_csv, args.select_k)[1])})")


if __name__ == "__main__":
    main()
