"""Mechanistic (fixed-model) gene attribution for the grid's best downstream head.

Decomposes a *linear* downstream classifier's decision back to genes through the
128-dim shared embedding space, without any leave-one-out retraining. Same three
stages as ``downstream_gene_attribution`` (0629), but:

* the downstream head is the **grid pipeline** ``imputer → scaler → selector →
  linear model`` (default ``Ridge``/``Variance`` k=85 — the Setting-A best cell /
  survival head A1), not the fixed LR(L1)/SelectKBest head; and
* no LOO / cosine-IG cross-comparison is produced (mechanistic importance only).

Stages
------
1. Fit the head on resection image embeddings vs ``rfs_2year`` and unwind it into a
   single decision direction ``beta ∈ R¹²⁸`` with ``logit(z) = beta·z + b``.
2. GeneEncoder Jacobian ``J[d,j] = ∂gene_enc(g)_d/∂g_j`` at the cohort-mean gene
   vector; ``C[d,j] = beta_d · J[d,j]`` is each gene's contribution to the decision
   axis per dim.
3. Integrated Gradients of ``s(g) = beta · gene_enc(g)`` w.r.t. the gene input,
   aggregated over resection patients as ``mean|IG|`` (importance) and ``mean IG``
   (signed; + = pushes toward recurrence ≤ 2yr).

Ensemble mode (``--members-csv``)
--------------------------------
With a ``model_ensemble_members.csv`` (written by ``embedding_grid_eval
--model-ensemble``) the head becomes the frozen top-k *model ensemble*
(:class:`~hcc_multimodal.eval.ensemble.HeteroEnsembleGrid`) — e.g. survival head A2.
That head is **not** a single decision direction: it mean-averages its members'
positive-class *probabilities*, so its score is

    S(z) = (1/M) Σ_m sigmoid( a_m (beta_m·z + b_m) + c_m )

with ``(a_m, c_m)`` the member's squash (identity for logistic regression, Platt for
``SVC(probability=True)``). Each member is still exactly linear, so stages 1–3 carry
over with the composition kept intact:

1. every member is unwound to its own ``(beta_m, b_m, a_m, c_m)``;
2. stage 2 uses the **local** direction ``beta_eff = ∇_z logit S(z)`` at the operating
   point ``z0 = gene_enc(mean gene vector)`` — a gradient-weighted mean of the members'
   directions — in place of the global ``beta``;
3. IG runs on ``s(g) = logit S(gene_enc(g))``, i.e. the ensemble's own
   ``decision_function``, so the attributed target is the deployed score itself and the
   units stay on the logit scale (comparable to the single-cell mode).

Requires a run with a deterministic gene order — either a refit (e.g. ``92ae6a23``, the
refit of ``dc7e1d10``), or a run trained after gene order started being persisted, whose
``metadata.json`` carries ``deterministic_gene_order`` plus its resolved ``gene_order``
(e.g. ``d7085bf5``); the latter attributes the co-trained gene branch directly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch
from scipy.special import logsumexp
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.utils import resample

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hcc_multimodal.eval.ensemble import HeteroEnsembleGrid, build_member
from hcc_multimodal.eval.grid import build_grid_pipeline, positive_scores
from hcc_multimodal.eval.metrics import compute_metrics
from hcc_multimodal.interpretability.downstream_gene_attribution import (
    _unwind_beta,
    gene_jacobian,
    integrated_gradients_beta,
    make_heatmap,
)
from hcc_multimodal.interpretability.gene_integrated_gradients import load_inputs
from hcc_multimodal.survival.data import load_embeddings, load_survival_outcomes
from hcc_multimodal.train.config import RANDOM_STATE


# ---------------------------------------------------------------------------
# Stage 1: fit the grid head and unwind the decision direction beta
# ---------------------------------------------------------------------------
def _make_head(fs: str, model: str, select_k: int, model_params: dict):
    pipe = build_grid_pipeline(fs, model, select_k)
    if model_params:
        pipe.set_params(**model_params)
    return pipe


def fit_downstream(model_id: str, fs: str, model: str, select_k: int, model_params: dict):
    """Fit the chosen grid head on resection embeddings vs ``rfs_2year`` and return
    (pipe, beta, bias, X_res, y_res, embed_dim)."""
    X = load_embeddings(model_id, "resection")
    y = load_survival_outcomes("resection")["rfs_2year"].dropna().astype(int)
    common = X.index.intersection(y.index)
    X, y = X.loc[common], y.loc[common]
    embed_dim = X.shape[1]

    pipe = _make_head(fs, model, select_k, model_params)
    pipe.fit(X, y)
    beta, bias = _unwind_beta(pipe, embed_dim)
    return pipe, beta, bias, X, y, embed_dim


def resection_cv_auc(X, y, fs, model, select_k, model_params) -> float:
    """Flat non-repeated 3-fold resection CV AUROC of the frozen head (matches the
    grid's selection metric for this cell)."""
    pipe = _make_head(fs, model, select_k, model_params)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    return float(cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc").mean())


def bootstrap_beta(X, y, embed_dim, fs, model, select_k, model_params, n_boot):
    """Refit the head on n_boot stratified patient resamples; return
    (mean_beta, sd_beta, selection_freq)."""
    betas = np.zeros((n_boot, embed_dim))
    selected = np.zeros((n_boot, embed_dim), dtype=bool)
    Xv, yv = X.values, y.values
    for b in range(n_boot):
        idx = resample(np.arange(len(yv)), replace=True, stratify=yv,
                       random_state=RANDOM_STATE + b)
        pipe = _make_head(fs, model, select_k, model_params)
        pipe.fit(Xv[idx], yv[idx])
        beta, _ = _unwind_beta(pipe, embed_dim)
        betas[b] = beta
        selected[b] = beta != 0.0
    return betas.mean(0), betas.std(0), selected.mean(0)


# ---------------------------------------------------------------------------
# Stage 1, ensemble mode: per-member directions + the composed score
# ---------------------------------------------------------------------------
def load_members(members_csv: Path) -> list[dict]:
    """Parse a ``model_ensemble_members.csv`` into frozen member specs."""
    df = pd.read_csv(members_csv)
    return [
        {
            "model": r["model"],
            "fs": r["fs"],
            "select_k": int(r["n_features_selected"]),
            "params": json.loads(r["best_params"]) if isinstance(r["best_params"], str) else {},
            "cv_auc": float(r["cv_auc_mean"]),
        }
        for _, r in df.iterrows()
    ]


def _make_ensemble(specs: list[dict]) -> HeteroEnsembleGrid:
    return HeteroEnsembleGrid(
        [build_member(s["model"], s["fs"], s["params"], s["select_k"]) for s in specs]
    )


def _squash_params(model) -> tuple[float, float]:
    """``(a, c)`` such that the member's positive-class score is
    ``sigmoid(a·(beta·z + b) + c)``.

    ``positive_scores`` takes ``predict_proba[:, 1]`` when available and
    ``expit(decision_function)`` otherwise. Logistic regression's proba is the identity
    squash of its own logit, and Ridge goes through the explicit ``expit``, so both are
    ``(1, 0)``. ``SVC(probability=True)`` fits Platt scaling on top of its decision
    function, ``p = 1/(1 + exp(A·f + B)) = sigmoid(−A·f − B)``.
    """
    if isinstance(model, SVC):
        return float(-model.probA_[0]), float(-model.probB_[0])
    return 1.0, 0.0


def unwind_ensemble(ens: HeteroEnsembleGrid, embed_dim: int) -> list[dict]:
    """Collapse each fitted member into ``(beta, bias, a, c)`` over the original dims."""
    out = []
    for m in ens.members_:
        beta, bias = _unwind_beta(m, embed_dim)
        a, c = _squash_params(m.named_steps["model"])
        out.append({"beta": beta, "bias": bias, "a": a, "c": c})
    return out


def _expit(u: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-u))


def member_logits(params: list[dict], Z: np.ndarray) -> np.ndarray:
    """Per-member squashed logits ``u_m = a_m(beta_m·z + b_m) + c_m``, shape (M, P)."""
    return np.array([p["a"] * (Z @ p["beta"] + p["bias"]) + p["c"] for p in params])


def ensemble_score(params: list[dict], Z: np.ndarray) -> np.ndarray:
    """Closed-form mean positive-class score ``S(z)`` of the unwound ensemble."""
    return _expit(member_logits(params, Z)).mean(axis=0)


def ensemble_logit(params: list[dict], Z: np.ndarray) -> np.ndarray:
    """``logit S(z)``, evaluated in the log domain so it stays exact under saturation.

    ``log S = logsumexp_m log σ_m − log M`` and ``log(1−S) = logsumexp_m log(1−σ_m) −
    log M``, so the ``log M`` cancels in the difference. This matters here: the head is
    fit on image embeddings, and gene embeddings run its members' scores ~50× wider, so
    the logistic members hit float 0/1 and a naive ``logit(mean sigmoid)`` returns ±inf
    (this is also where :meth:`HeteroEnsembleGrid.decision_function`'s 1e-9 clip bites).
    The identity below is the unclipped, mathematically exact same quantity.
    """
    u = member_logits(params, Z)
    log_s = -np.logaddexp(0.0, -u)      # log sigmoid(u)
    log_1ms = -np.logaddexp(0.0, u)     # log (1 - sigmoid(u))
    return logsumexp(log_s, axis=0) - logsumexp(log_1ms, axis=0)


def _grad_weights(params: list[dict], Z: np.ndarray) -> np.ndarray:
    """Per-member weight ``w_m = a_m σ_m(1−σ_m) / [M·S(1−S)]`` in ``∇_z logit S =
    Σ_m w_m beta_m``, shape (M, P). Formed in the log domain for the saturation reason
    documented in :func:`ensemble_logit`."""
    u = member_logits(params, Z)                           # (M, P)
    log_s = -np.logaddexp(0.0, -u)
    log_1ms = -np.logaddexp(0.0, u)
    log_dsig = log_s + log_1ms                             # log σ_m(1−σ_m)
    # log[ M·S(1−S) ] = logsumexp(log σ) + logsumexp(log(1−σ)) − log M
    log_denom = logsumexp(log_s, axis=0) + logsumexp(log_1ms, axis=0) - np.log(len(params))
    return np.array([p["a"] for p in params])[:, None] * np.exp(log_dsig - log_denom)


def gradient_shares(params: list[dict], Z: np.ndarray) -> np.ndarray:
    """Each member's share of the summed gradient magnitude ``|w_m|·‖beta_m‖``, averaged
    over the rows of ``Z`` — how much of the ensemble score's movement it accounts for at
    that operating point. Saturated members carry ~0 share."""
    mag = np.abs(_grad_weights(params, Z)) * np.array(
        [np.linalg.norm(p["beta"]) for p in params]
    )[:, None]
    mean_mag = mag.mean(axis=1)
    total = mean_mag.sum()
    return mean_mag / total if total > 0 else mean_mag


def beta_effective(params: list[dict], z0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Local decision direction of the ensemble at operating point ``z0``.

    ``S(z) = mean_m sigmoid(a_m(beta_m·z + b_m) + c_m)`` is not linear, so stage 2 uses
    the gradient of the ensemble's own ``decision_function = logit S`` at ``z0``, a
    gradient-weighted mean of the member directions. Also returns the per-member
    gradient shares there.
    """
    w = _grad_weights(params, z0[None, :])[:, 0]           # (M,)
    beta_eff = (w[:, None] * np.array([p["beta"] for p in params])).sum(axis=0)
    return beta_eff, gradient_shares(params, z0[None, :])


def fit_downstream_ensemble(model_id: str, specs: list[dict]):
    """Fit the frozen model ensemble on resection embeddings vs ``rfs_2year``; return
    (ens, params, X_res, y_res, embed_dim)."""
    X = load_embeddings(model_id, "resection")
    y = load_survival_outcomes("resection")["rfs_2year"].dropna().astype(int)
    common = X.index.intersection(y.index)
    X, y = X.loc[common], y.loc[common]
    embed_dim = X.shape[1]

    ens = _make_ensemble(specs)
    ens.fit(X, y)
    return ens, unwind_ensemble(ens, embed_dim), X, y, embed_dim


def resection_cv_auc_ensemble(X, y, specs) -> float:
    """Flat non-repeated 3-fold resection CV AUROC of the frozen ensemble (matches the
    grid's ``model_ensemble_best.csv`` selection metric)."""
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    return float(cross_val_score(_make_ensemble(specs), X, y, cv=cv, scoring="roc_auc").mean())


def bootstrap_beta_eff(X, y, embed_dim, specs, z0, n_boot):
    """Refit the whole ensemble on ``n_boot`` stratified patient resamples and re-read
    ``beta_eff`` at the fixed operating point ``z0``; return
    (mean, sd, nonzero_freq). ``z0`` depends only on the gene branch, so it is held
    fixed across resamples and only the head varies."""
    betas = np.zeros((n_boot, embed_dim))
    nonzero = np.zeros((n_boot, embed_dim), dtype=bool)
    Xv, yv = X.values, y.values
    for b in range(n_boot):
        idx = resample(np.arange(len(yv)), replace=True, stratify=yv,
                       random_state=RANDOM_STATE + b)
        ens = _make_ensemble(specs)
        ens.fit(Xv[idx], yv[idx])
        beta_eff, _ = beta_effective(unwind_ensemble(ens, embed_dim), z0)
        betas[b] = beta_eff
        nonzero[b] = beta_eff != 0.0
    return betas.mean(0), betas.std(0), nonzero.mean(0)


def transfer_auroc(pipe, model_id: str, cohort: str) -> float:
    """Resection-fit head → external-cohort AUROC (uses positive_scores so Ridge's
    proba-less decision_function is handled)."""
    X = load_embeddings(model_id, cohort)
    y = load_survival_outcomes(cohort)["rfs_2year"].dropna().astype(int)
    common = X.index.intersection(y.index)
    scores = positive_scores(pipe, X.loc[common])
    return compute_metrics(y.loc[common].values, scores)["auroc"]


# ---------------------------------------------------------------------------
# Stage 3, ensemble mode: IG on s(g) = logit S(gene_enc(g))
# ---------------------------------------------------------------------------
def integrated_gradients_ensemble(gene_enc, params, x, baseline, steps, device):
    """Midpoint-rule IG of the ensemble's own decision function

        s(g) = logit( (1/M) Σ_m sigmoid(a_m(beta_m·gene_enc(g) + b_m) + c_m) )

    w.r.t. the gene input. Same contract as
    :func:`~hcc_multimodal.interpretability.downstream_gene_attribution.integrated_gradients_beta`:
    returns (ig (P,G), residual (P,), delta_s (P,)), with the completeness residual
    ``sum_j ig − (s(x) − s(baseline))`` expected ~0.

    The target is evaluated in the log domain (see :func:`ensemble_logit`) and in double
    precision: on the gene branch the members' scores run far into saturation, where a
    literal ``logit(mean sigmoid)`` is ±inf and its gradient is NaN. This is the same
    quantity :meth:`HeteroEnsembleGrid.decision_function` returns, without its 1e-9 clip.
    """
    t = lambda v: torch.tensor(v, dtype=torch.float64, device=device)  # noqa: E731
    B = t(np.array([p["beta"] for p in params]))          # (M, D)
    b0 = t(np.array([p["bias"] for p in params]))         # (M,)
    a = t(np.array([p["a"] for p in params]))             # (M,)
    c = t(np.array([p["c"] for p in params]))             # (M,)

    P, G = x.shape
    base = baseline.unsqueeze(0).expand(P, G)
    total_grad = torch.zeros_like(x)

    def s(g):
        z = gene_enc(g).double()                           # (P, D)
        u = a * (z @ B.T + b0) + c                         # (P, M)
        # log S − log(1−S); the −log M of each mean cancels in the difference
        return (torch.logsumexp(-torch.nn.functional.softplus(-u), dim=-1)
                - torch.logsumexp(-torch.nn.functional.softplus(u), dim=-1))

    alphas = (torch.arange(steps, device=device).float() + 0.5) / steps
    for alpha in alphas:
        g = (base + alpha * (x - base)).clone().requires_grad_(True)
        s(g).sum().backward()
        total_grad += g.grad

    avg_grad = total_grad / steps
    ig = (x - base) * avg_grad

    with torch.no_grad():
        delta_s = s(x) - s(base)
    residual = ig.sum(dim=-1) - delta_s
    return ig.detach(), residual.detach(), delta_s.detach()


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def _report_text(args, meta, ensemble_mode: bool, members) -> dict:
    """Mode-dependent prose + label fragments for :func:`write_report`."""
    if meta.get("refit_gene_enc_from"):
        src = meta["refit_gene_enc_from"]
        provenance = (
            f"**Model provenance.** `{args.model_id}` is a refit "
            f"(`scripts`/`hcc_multimodal.interpretability.refit_gene_encoder`, "
            f"`--precompute-embeddings`) of `{src}`: the image encoder and its cached "
            f"embeddings are reused **verbatim** (resection embedding cache is "
            f"byte-identical), so the downstream head is identical to `{src}`'s by "
            f"construction. Only the GeneEncoder is re-aligned to that frozen image space "
            f"under a pinned, sorted gene order, giving an interpretable gene branch."
        )
        caveat_tail = "on a re-aligned gene branch."
    else:
        provenance = (
            f"**Model provenance.** `{args.model_id}` is attributed **directly** — no gene-encoder "
            f"refit. It was trained with its resolved gene column order persisted "
            f"(`deterministic_gene_order`, `gene_order` in `metadata.json`), so the trained "
            f"GeneEncoder's gene→slot mapping is recoverable and the co-trained gene branch is "
            f"itself interpretable. Image encoder, gene encoder and the cached embeddings the "
            f"head is fit on all come from the same `best_model.pt` checkpoint."
        )
        caveat_tail = "on the co-trained gene branch."

    if ensemble_mode:
        cells = ", ".join(f"`{m['model']}`/`{m['fs']}` k={m['select_k']}" for m in members)
        method = (
            f"**Method.** (1) The head is the frozen top-{len(members)} **model ensemble** "
            f"({cells}), each member a `SimpleImputer(median) → StandardScaler → selector → "
            f"linear model` pipeline fit on the {{n_res}} resection patients' image embeddings vs "
            f"`rfs_2year`. Every member is unwound exactly into its own direction "
            f"`β_m ∈ R¹²⁸` and squash, so the ensemble score is "
            f"`S(z) = (1/M) Σ_m sigmoid(a_m(β_m·z + b_m) + c_m)` — a mean over member "
            f"*probabilities*, hence **not** a single decision direction. (2) The GeneEncoder "
            f"Jacobian `J[d,j] = ∂geneenc(g)_d/∂g_j` (at the cohort-mean gene vector) gives each "
            f"gene's effect on each shared dim; the decision axis is the **local** direction "
            f"`β_eff = ∇_z logit S` at the operating point `z0 = geneenc(ḡ)` — a "
            f"gradient-weighted mean of the `β_m` — and `C[d,j] = β_eff,d·J[d,j]`. "
            f"(3) Integrated Gradients of `s(g) = logit S(geneenc(g))` — the ensemble's own "
            f"`decision_function`, i.e. the deployed score on the logit scale — w.r.t. the gene "
            f"input ({args.steps} midpoint steps, baseline = `{args.baseline}`), aggregated over "
            f"{{n_patients}} patients as `mean|IG|` (importance) and `mean IG` (signed; "
            f"**+ = pushes toward recurrence ≤ 2yr**)."
        )
        recon_label = (
            "Ensemble-score reconstruction check (max |closed-form S − `predict_proba`|; "
            "gap is libsvm's iterative pairwise-coupling step in the Platt member, not the "
            "unwinding)"
        )
        beta_sym, freq_label = "β_eff", "nonzero freq"
        freq_desc = ("the dim carries nonzero weight in β_eff — i.e. some member both selects "
                     "it and is not fully saturated at `z0`")
    else:
        method = (
            f"**Method.** (1) The grid head (`SimpleImputer(median) → StandardScaler → "
            f"{args.fs}(k={args.select_k}) → {args.model}`{{params_str}}) is fit on the {{n_res}} "
            f"resection patients' image embeddings vs `rfs_2year`, then collapsed into a single "
            f"decision direction `β ∈ R¹²⁸` with `logit(z) = β·z + b`. (2) The GeneEncoder "
            f"Jacobian `J[d,j] = ∂geneenc(g)_d/∂g_j` (at the cohort-mean gene vector) gives each "
            f"gene's effect on each shared dim; `C[d,j] = β_d·J[d,j]` is its contribution to the "
            f"decision axis. (3) Integrated Gradients of `s(g) = β·geneenc(g)` w.r.t. the gene "
            f"input ({args.steps} midpoint steps, baseline = `{args.baseline}`) aggregated over "
            f"{{n_patients}} patients as `mean|IG|` (importance) and `mean IG` (signed; "
            f"**+ = pushes toward recurrence ≤ 2yr**)."
        )
        recon_label = "β reconstruction check (max |β·z+b − decision_function|)"
        beta_sym, freq_label = "β", "sel. freq"
        freq_desc = "the selector keeps the dim"

    caveat = (
        "> **Caveat.** Genes never enter the predictor at inference — they shape the image "
        "encoder only through the contrastive alignment at training time. This is an "
        "*alignment-mediated proxy*: the per-dimension decomposition of the decision axis "
        + caveat_tail
    )
    return {"provenance": provenance, "method": method, "caveat": caveat,
            "recon_label": recon_label, "beta_sym": beta_sym,
            "freq_label": freq_label, "freq_desc": freq_desc}


def write_report(ctx, report_path: Path):
    df = ctx["df"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    L = []
    L.append(f"# Mechanistic Gene Attribution — {pd.Timestamp.today():%Y-%m-%d}")
    L.append("")
    L.append(
        f"Attributes the **{ctx['head']}** downstream RFS classifier's decision back to "
        f"the {ctx['n_genes']} genes of `{ctx['gene_set']}` for run `{ctx['model_id']}`, by "
        f"decomposing it through the 128-dim shared embedding space. Mechanistic "
        f"(fixed-model) importance only — no leave-one-out retraining."
    )
    L.append("")
    L.append(ctx["provenance"])
    L.append("")
    L.append(ctx["method"].format(n_res=ctx["n_res"], n_patients=ctx["n_patients"],
                                  params_str=ctx["params_str"]))
    L.append("")
    L.append(ctx["caveat"])
    L.append("")
    L.append(f"- Completeness check (max |Σ_j IG − (s(x)−s(baseline))|): **{ctx['max_resid']:.2e}** "
             f"= **{ctx['rel_resid']:.2e}** of the per-patient target range")
    L.append(f"- {ctx['recon_label']}: **{ctx['beta_recon']:.2e}**")
    L.append(f"- Resection 3-fold CV AUROC of the head: **{ctx['cv_auc']:.3f}** "
             f"({ctx['cv_ref']})")
    L.append(f"- Soramic transfer AUROC: **{ctx['soramic']:.3f}** ({ctx['soramic_ref']}); "
             f"Lausanne: **{ctx['lusanne']:.3f}**")
    L.append("")

    # Member table (ensemble mode)
    if ctx.get("members"):
        L.append("## 0. Ensemble members")
        L.append("")
        L.append("`‖β_m‖` and `nnz` describe each member's own decision direction over the 128 "
                 "dims. `squash a` is the slope of its probability squash (1 = plain logistic, "
                 "Platt-fitted for L-SVM); a flat squash compresses that member's score range and "
                 "so shrinks its weight in the probability average. `grad share` is the member's "
                 "fraction of the summed gradient magnitude — how much it actually moves the "
                 "ensemble score — reported at **both** operating points: the resection *image* "
                 "embeddings the head is fit on and scores, and the *gene* embedding `z0` that "
                 "stages 2–3 push through it.")
        L.append("")
        L.append("| Member | FS | k | CV AUC | ‖β_m‖ | nnz | squash a | u(image) | u(gene) | "
                 "grad share (image) | grad share (gene) |")
        L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for m in ctx["members"]:
            ri, rg = m["squashed_logit_range_image"], m["squashed_logit_range_gene"]
            L.append(f"| {m['model']} | {m['fs']} | {m['select_k']} | {m['cv_auc']:.3f} | "
                     f"{m['beta_norm']:.1f} | {m['beta_nonzero']} | {m['squash_a']:.3f} | "
                     f"[{ri[0]:.1f}, {ri[1]:.1f}] | [{rg[0]:.0f}, {rg[1]:.0f}] | "
                     f"{m['gradient_share_image']:.3f} | {m['gradient_share_gene']:.3f} |")
        L.append("")
        L.append("`u` is the member's squashed logit `a_m(β_m·z + b_m) + c_m` — the argument of "
                 "its sigmoid — over each cohort of embeddings. A member only responds where |u| "
                 "is small: |u| = 4 leaves only ~7% of its peak gradient, and |u| ≳ 20 is "
                 "numerically total saturation.")
        L.append("")

    # Table 1
    L.append("## 1. Per-gene mechanistic importance (sorted by `mean|IG|`)")
    L.append("")
    L.append("| Gene | mean\\|IG\\| | signed mean IG | rank |")
    L.append("|---|---:|---:|---:|")
    ig_rank = df.sort_values("mean_abs_ig", ascending=False).reset_index(drop=True)
    for i, r in ig_rank.iterrows():
        L.append(f"| {r['gene']} | {r['mean_abs_ig']:.4f} | {r['signed_mean_ig']:+.4f} | {i + 1} |")
    L.append("")
    L.append(f"![Gene → decision-axis contribution heatmap]({ctx['heatmap_name']})")
    L.append("")

    # Table 2
    sym = ctx["beta_sym"]
    L.append(f"## 2. Top downstream dimensions by |{sym}| (with bootstrap stability)")
    L.append("")
    L.append(f"Bootstrap = {ctx['n_boot']} stratified patient resamples. `{ctx['freq_label']}` = "
             f"fraction of resamples where {ctx['freq_desc']}. `top driver genes` = "
             f"largest |C[d,j]|.")
    L.append("")
    L.append(f"| dim | {sym} | bootstrap mean±sd | {ctx['freq_label']} | "
             f"top driver genes (signed C) |")
    L.append("|---|---:|---:|---:|---|")
    for d in ctx["top_dims"]:
        drv = ", ".join(f"{g} ({c:+.3f})" for g, c in ctx["dim_drivers"][d])
        L.append(f"| {d} | {ctx['beta'][d]:+.3f} | {ctx['boot_mean'][d]:+.3f}±{ctx['boot_sd'][d]:.3f} | "
                 f"{ctx['boot_freq'][d]:.2f} | {drv} |")
    L.append("")
    L.append("**Notes**")
    L.append("")
    L.append(f"- Stage-1 {sym} says *which embedding dims the classifier uses*; stage-2 C and "
             "stage-3 IG say *which genes move the patient along those dims*. A gene ranks high "
             "only if it drives dims the classifier weights.")
    L.append("- Signed IG > 0 ⇒ higher expression pushes the patient toward 2-year recurrence along "
             "the classifier's decision axis.")
    report_path.write_text("\n".join(L) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_params = json.loads(args.model_params) if args.model_params else {}
    ensemble_mode = args.members_csv is not None

    # Stage 1: the downstream head
    specs = params = member_share = None
    bias = None
    if ensemble_mode:
        specs = load_members(args.members_csv)
        head_obj, params, X_res, y_res, embed_dim = fit_downstream_ensemble(args.model_id, specs)
        # the composed closed form vs the estimator's own averaged predict_proba
        beta_recon = float(np.max(np.abs(
            ensemble_score(params, X_res.values) - head_obj.predict_proba(X_res)[:, 1]
        )))
        cv_auc = resection_cv_auc_ensemble(X_res, y_res, specs)
        cells = " + ".join(f"{s['model']}/{s['fs']} k={s['select_k']}" for s in specs)
        head = f"top-{len(specs)} model ensemble ({cells})"
    else:
        head_obj, beta_np, bias, X_res, y_res, embed_dim = fit_downstream(
            args.model_id, args.fs, args.model, args.select_k, model_params
        )
        beta_recon = float(np.max(np.abs(
            X_res.values @ beta_np + bias - head_obj.decision_function(X_res)
        )))
        cv_auc = resection_cv_auc(X_res, y_res, args.fs, args.model, args.select_k, model_params)
        head = f"{args.model}/{args.fs} k={args.select_k}"
    soramic = transfer_auroc(head_obj, args.model_id, "soramic")
    lusanne = transfer_auroc(head_obj, args.model_id, "lusanne")

    # Gene branch
    gene_enc, gene_names, x, _z_img, meta, pids = load_inputs(args.model_id, device)
    g_bar = x.mean(0)

    # Stage 2: Jacobian at the cohort-mean gene vector. The ensemble is not a single
    # direction, so its decision axis is the local gradient at that same operating point.
    J = gene_jacobian(gene_enc, g_bar)
    if ensemble_mode:
        with torch.no_grad():
            z0 = gene_enc(g_bar.unsqueeze(0)).squeeze(0).cpu().numpy()
        beta_np, member_share = beta_effective(params, z0)
        boot_mean, boot_sd, boot_freq = bootstrap_beta_eff(
            X_res, y_res, embed_dim, specs, z0, args.n_boot
        )
    else:
        boot_mean, boot_sd, boot_freq = bootstrap_beta(
            X_res, y_res, embed_dim, args.fs, args.model, args.select_k, model_params, args.n_boot
        )
    C = beta_np[:, None] * J
    top_dims = [int(d) for d in np.argsort(-np.abs(beta_np))[: args.top_k] if beta_np[d] != 0.0]
    dim_drivers = {}
    for d in top_dims:
        order = np.argsort(-np.abs(C[d]))[:3]
        dim_drivers[d] = [(gene_names[j], float(C[d, j])) for j in order]

    # Stage 3: IG
    baseline = torch.zeros(x.shape[1], device=device) if args.baseline == "zero" else g_bar
    if ensemble_mode:
        ig, residual, delta_s = integrated_gradients_ensemble(
            gene_enc, params, x, baseline, args.steps, device
        )
    else:
        beta = torch.tensor(beta_np, dtype=torch.float32, device=device)
        ig, residual, delta_s = integrated_gradients_beta(
            gene_enc, beta, x, baseline, args.steps, device
        )
    max_resid = float(residual.abs().max().cpu())
    rel_resid = float((residual.abs().max() / delta_s.abs().max().clamp_min(1e-12)).cpu())

    df = pd.DataFrame({
        "gene": gene_names,
        "mean_abs_ig": ig.abs().mean(0).cpu().numpy(),
        "signed_mean_ig": ig.mean(0).cpu().numpy(),
    })

    params_str = f", α={model_params['model__alpha']}" if "model__alpha" in model_params else ""

    # Figure
    args.report.parent.mkdir(parents=True, exist_ok=True)
    heatmap_path = args.report.parent / (args.report.stem + "_heatmap.png")
    make_heatmap(C, gene_names, top_dims, heatmap_path)

    # Per-member summary (ensemble mode only)
    members_out = None
    if ensemble_mode:
        # Contrast the two operating points: the image embeddings the head is fit on and
        # actually scores, vs the gene embeddings IG runs through.
        share_img = gradient_shares(params, X_res.values)
        with torch.no_grad():
            z_gene = gene_enc(x).cpu().numpy()
        f_img = member_logits(params, X_res.values)
        f_gene = member_logits(params, z_gene)
        members_out = [
            {
                "model": s["model"], "fs": s["fs"], "select_k": s["select_k"],
                "cv_auc": s["cv_auc"], "params": s["params"],
                "squash_a": p["a"], "squash_c": p["c"],
                "beta_norm": float(np.linalg.norm(p["beta"])),
                "beta_nonzero": int((p["beta"] != 0).sum()),
                "gradient_share_image": float(si),
                "gradient_share_gene": float(sg),
                "squashed_logit_range_image": [float(fi.min()), float(fi.max())],
                "squashed_logit_range_gene": [float(fg.min()), float(fg.max())],
            }
            for s, p, si, sg, fi, fg in zip(
                specs, params, share_img, member_share, f_img, f_gene
            )
        ]

    # JSON
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "model_id": args.model_id, "refit_from": meta.get("refit_gene_enc_from"),
        "gene_set": meta["gene_set"], "head": head, "model_params": model_params,
        "ensemble_mode": ensemble_mode, "members": members_out,
        "baseline": args.baseline, "steps": args.steps, "n_patients": len(pids),
        "n_genes": len(df), "n_boot": args.n_boot,
        "cv_auc": cv_auc, "soramic_auroc": soramic, "lusanne_auroc": lusanne,
        "completeness_max_residual": max_resid, "completeness_max_residual_relative": rel_resid,
        "beta_reconstruction_max_residual": beta_recon,
        "beta": beta_np.tolist(), "bias": bias,
        "bootstrap_beta_mean": boot_mean.tolist(), "bootstrap_beta_sd": boot_sd.tolist(),
        "bootstrap_selection_freq": boot_freq.tolist(),
        "top_dims": top_dims, "dim_drivers": {str(d): dim_drivers[d] for d in top_dims},
        "genes": df.sort_values("mean_abs_ig", ascending=False).to_dict(orient="records"),
    }, indent=2))
    print(f"Wrote results → {args.output}")

    ctx = dict(
        df=df, model_id=args.model_id, refit_from=meta.get("refit_gene_enc_from"),
        gene_set=meta["gene_set"], head=head, fs=args.fs, model=args.model,
        select_k=args.select_k, params_str=params_str,
        n_genes=len(df), n_patients=len(pids), n_res=len(X_res),
        steps=args.steps, baseline=args.baseline, n_boot=args.n_boot,
        cv_auc=cv_auc, soramic=soramic, lusanne=lusanne,
        cv_ref=args.ref_cv_label, soramic_ref=args.ref_soramic_label,
        max_resid=max_resid, rel_resid=rel_resid, beta_recon=beta_recon,
        beta=beta_np, boot_mean=boot_mean, boot_sd=boot_sd, boot_freq=boot_freq,
        top_dims=top_dims, dim_drivers=dim_drivers, heatmap_name=heatmap_path.name,
        members=members_out,
        **_report_text(args, meta, ensemble_mode, members_out),
    )
    write_report(ctx, args.report)
    print(f"Wrote report → {args.report}")

    print(f"\nHead: {head}{params_str}")
    print(f"Resection CV AUROC: {cv_auc:.3f} | Soramic: {soramic:.3f} | Lausanne: {lusanne:.3f}")
    if members_out:
        for m in members_out:
            print(f"  member {m['model']}/{m['fs']} k={m['select_k']}: "
                  f"CV {m['cv_auc']:.3f} | ‖β‖ {m['beta_norm']:.1f} | "
                  f"nnz {m['beta_nonzero']} | grad share img {m['gradient_share_image']:.3f} "
                  f"/ gene {m['gradient_share_gene']:.3f}")
    print(f"{'score' if ensemble_mode else 'β'} reconstruction max residual: {beta_recon:.2e}")
    print(f"IG completeness max residual:  {max_resid:.2e} ({rel_resid:.2e} relative)")
    print("\nTop genes by mechanistic mean|IG|:")
    print(df.sort_values("mean_abs_ig", ascending=False)[
        ["gene", "mean_abs_ig", "signed_mean_ig"]
    ].head(10).to_string(index=False))
    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-id", default="92ae6a23", help="Refit run id (deterministic gene order)")
    p.add_argument("--fs", default="Variance", help="Feature-selection name (grid FS_ZOO)")
    p.add_argument("--model", default="Ridge", help="Classifier name (grid MODEL_ZOO); must be linear")
    p.add_argument("--select-k", type=int, default=85)
    p.add_argument("--model-params", default='{"model__alpha": 100.0}',
                   help="JSON of pipeline params, e.g. tuned hyperparameters for this cell")
    p.add_argument("--members-csv", type=Path, default=None,
                   help="model_ensemble_members.csv from embedding_grid_eval --model-ensemble. "
                        "When given, the head is that frozen model ensemble and --fs/--model/"
                        "--select-k/--model-params are ignored.")
    p.add_argument("--ref-cv-label", default="dc7e1d10 grid best cell = 0.744",
                   help="Parenthetical reference for the CV AUROC bullet")
    p.add_argument("--ref-soramic-label", default="dc7e1d10 best cell = 0.709",
                   help="Parenthetical reference for the Soramic AUROC bullet")
    p.add_argument("--baseline", choices=["zero", "mean"], default="zero")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--n-boot", type=int, default=500)
    p.add_argument("--top-k", type=int, default=15)
    p.add_argument("--output", type=Path,
                   default=Path("results/eval/soramic/gene_ablation/mechanistic_gene_attribution.json"))
    p.add_argument("--report", type=Path,
                   default=Path("reports/0727/0727_mechanistic_interpretability.md"))
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
