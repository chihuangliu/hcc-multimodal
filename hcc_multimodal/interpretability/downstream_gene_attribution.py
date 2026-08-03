"""Downstream-classifier → gene attribution for a contrastive image-gene model.

Decomposes the RFS classifier's decision back to genes in three stages:

1. **Downstream feature importance.** Fit the exact downstream pipeline
   (``SimpleImputer → StandardScaler → SelectKBest → LR(L1)``) on the resection
   image embeddings and ``rfs_2year``, then *unwind* it into a single decision
   direction ``beta`` over the original 128 embedding dims:

       logit(z) = beta . z + b,   beta_d = lr.coef_[j] / scaler.scale_[j]

   for selected dim d (0 otherwise). A bootstrap over patient resamples gives
   per-dim stability (mean/sd of beta_d, selection frequency).

2. **Gene → shared-dim Jacobian.** ``J[d, j] = d gene_enc(g)_d / d g_j`` at the
   cohort-mean gene vector. Per-(gene, dim) contribution to the decision axis is
   ``C[d, j] = beta_d * J[d, j]``.

3. **Composite per-gene importance.** Integrated Gradients of the population
   decision target

       s(g) = beta . gene_enc(g)

   w.r.t. the gene input, aggregated over resection patients as ``mean|IG|``
   (importance) and ``mean IG`` (signed; + = pushes toward recurrence ≤ 2yr).

The gene branch reuses ``gene_integrated_gradients.load_inputs`` (which enforces a
deterministic gene order), so this must be run on a refit model such as
``a0912600`` (image encoder identical to ``9109a6c2``).

Caveat: genes never enter the predictor at inference; they shape the image
encoder only through the contrastive alignment at train time. This attribution is
therefore an *alignment-mediated proxy* — the per-dim decomposition of the
whole-vector cosine target used in ``gene_integrated_gradients.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.utils import resample

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hcc_multimodal.eval.eval_utils import DOWNSTREAM_MODELS, build_pipeline
from hcc_multimodal.eval.metrics import compute_metrics
from hcc_multimodal.interpretability.gene_integrated_gradients import (
    load_inputs,
    load_loo,
)
from hcc_multimodal.survival.data import load_embeddings, load_survival_outcomes
from hcc_multimodal.train.config import RANDOM_STATE

SELECT_K = 100


# ---------------------------------------------------------------------------
# Stage 1: downstream decision direction beta
# ---------------------------------------------------------------------------
def _unwind_beta(pipe, embed_dim: int) -> tuple[np.ndarray, float]:
    """Collapse the fitted (scaler → SelectKBest → LR) pipeline into a single
    linear decision direction over the original `embed_dim` features:
    logit(z) = beta . z + b. The median imputer is a no-op on NaN-free
    embeddings (it only shifts the bias, which we recompute exactly below)."""
    scaler = pipe.named_steps["scaler"]
    selector = pipe.named_steps["selector"]
    lr = pipe.named_steps["model"]

    mask = selector.get_support()  # (embed_dim,), True for selected dims
    sel_idx = np.flatnonzero(mask)
    coef = lr.coef_.ravel()  # (n_selected,)
    scale = scaler.scale_[mask]  # (n_selected,)
    mean = scaler.mean_[mask]

    beta = np.zeros(embed_dim)
    beta[sel_idx] = coef / scale
    # logit = sum_j coef_j * (z_sel_j - mean_j)/scale_j + intercept
    bias = float(lr.intercept_.ravel()[0] - np.sum(coef * mean / scale))
    return beta, bias


def fit_downstream(model_id: str):
    """Fit the headline LR(L1) downstream classifier on resection embeddings and
    return (pipe, beta, bias, X_res, y_res, embed_dim)."""
    X = load_embeddings(model_id, "resection")
    y = load_survival_outcomes("resection")["rfs_2year"].dropna().astype(int)
    common = X.index.intersection(y.index)
    X, y = X.loc[common], y.loc[common]
    embed_dim = X.shape[1]

    pipe = build_pipeline(DOWNSTREAM_MODELS["lr"], min(SELECT_K, embed_dim))
    pipe.fit(X, y)
    beta, bias = _unwind_beta(pipe, embed_dim)
    return pipe, beta, bias, X, y, embed_dim


def bootstrap_beta(X: pd.DataFrame, y: pd.Series, embed_dim: int, n_boot: int):
    """Refit the downstream pipeline on `n_boot` stratified patient resamples;
    return (mean_beta, sd_beta, selection_freq) over the original dims."""
    betas = np.zeros((n_boot, embed_dim))
    selected = np.zeros((n_boot, embed_dim), dtype=bool)
    Xv, yv = X.values, y.values
    for b in range(n_boot):
        idx = resample(
            np.arange(len(yv)), replace=True, stratify=yv, random_state=RANDOM_STATE + b
        )
        pipe = build_pipeline(DOWNSTREAM_MODELS["lr"], min(SELECT_K, embed_dim))
        pipe.fit(Xv[idx], yv[idx])
        beta, _ = _unwind_beta(pipe, embed_dim)
        betas[b] = beta
        selected[b] = beta != 0.0
    return betas.mean(0), betas.std(0), selected.mean(0)


def sanity_soramic_auroc(pipe, model_id: str) -> float:
    """Resection-fit pipeline → Soramic AUROC; must reproduce the headline 0.732."""
    X_abl = load_embeddings(model_id, "soramic")
    y_abl = load_survival_outcomes("soramic")["rfs_2year"].dropna().astype(int)
    common = X_abl.index.intersection(y_abl.index)
    proba = pipe.predict_proba(X_abl.loc[common])[:, 1]
    return compute_metrics(y_abl.loc[common].values, proba)["auroc"]


# ---------------------------------------------------------------------------
# Stage 2: gene -> shared-dim Jacobian
# ---------------------------------------------------------------------------
def gene_jacobian(gene_enc, g_point: torch.Tensor) -> np.ndarray:
    """Jacobian d gene_enc(g)_d / d g_j at a single gene vector g_point (G,).
    Returns (embed_dim, gene_dim)."""
    jac = torch.autograd.functional.jacobian(
        lambda g: gene_enc(g.unsqueeze(0)).squeeze(0), g_point
    )
    return jac.detach().cpu().numpy()  # (embed_dim, gene_dim)


# ---------------------------------------------------------------------------
# Stage 3: Integrated Gradients on the decision target s(g) = beta . gene_enc(g)
# ---------------------------------------------------------------------------
def integrated_gradients_beta(gene_enc, beta, x, baseline, steps, device):
    """Midpoint-rule IG of s(g) = beta . gene_enc(g) w.r.t. the gene input.

    Returns (ig (P,G), residual (P,), delta_s (P,), avg_grad (P,G)). The completeness
    residual sum_j ig - (s(x) - s(baseline)) should be ~0. ``avg_grad`` is the path
    average of ds/dg_j, i.e. the ``ig = (x - baseline) * avg_grad`` factor that carries
    the sensitivity without the input-offset lever arm.
    """
    P, G = x.shape
    base = baseline.unsqueeze(0).expand(P, G)
    total_grad = torch.zeros_like(x)

    def s(g):
        return (gene_enc(g) * beta).sum(dim=-1)  # (P,)

    alphas = (torch.arange(steps, device=device).float() + 0.5) / steps
    for a in alphas:
        g = (base + a * (x - base)).clone().requires_grad_(True)
        s(g).sum().backward()
        total_grad += g.grad

    avg_grad = total_grad / steps
    ig = (x - base) * avg_grad

    with torch.no_grad():
        delta_s = s(x) - s(base)
    residual = ig.sum(dim=-1) - delta_s
    return ig.detach(), residual.detach(), delta_s.detach(), avg_grad.detach()


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def make_heatmap(C: np.ndarray, gene_names, top_dims, fig_path: Path):
    """Heatmap of C[d, j] = beta_d * J[d, j] over (top dims x genes)."""
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    sub = C[top_dims]  # (K, G)
    vmax = np.abs(sub).max()
    fig, ax = plt.subplots(figsize=(0.5 * len(gene_names) + 2, 0.4 * len(top_dims) + 2))
    im = ax.imshow(sub, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(gene_names)))
    ax.set_xticklabels(gene_names, rotation=90, fontsize=7)
    ax.set_yticks(range(len(top_dims)))
    ax.set_yticklabels([f"dim {d}" for d in top_dims], fontsize=7)
    ax.set_title("Gene → decision-axis contribution  C[d,j] = β_d · ∂geneenc_d/∂g_j")
    fig.colorbar(im, ax=ax, fraction=0.025, label="contribution (+ → recurrence)")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def make_scatter(df: pd.DataFrame, fig_path: Path):
    """Composite-IG vs cosine-IG gene importance, if cosine IG is available."""
    if "cos_ig" not in df or df["cos_ig"].isna().all():
        return
    sub = df.dropna(subset=["cos_ig"])
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(sub["cos_ig"], sub["mean_abs_ig"], s=30)
    for _, r in sub.iterrows():
        ax.annotate(r["gene"], (r["cos_ig"], r["mean_abs_ig"]), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("cosine-IG importance  mean|IG|  (alignment target)")
    ax.set_ylabel("downstream-IG importance  mean|IG|  (decision target)")
    ax.set_title("Downstream-decision vs cosine-alignment gene importance")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def make_quadrant(df: pd.DataFrame, fig_path: Path):
    """LOO vs downstream-IG four-quadrant plot. x = LOO importance (−ΔAUC),
    y = downstream mean|IG|; median lines split both-high / mechanistic-only /
    LOO-only / both-low. Returns the per-quadrant gene lists for the report."""
    sub = df.dropna(subset=["loo_delta"]).copy()
    sub["loo_imp"] = -sub["loo_delta"]  # more positive = more important
    xm, ym = sub["loo_imp"].median(), sub["mean_abs_ig"].median()

    quad = {"both_high": [], "mech_only": [], "loo_only": [], "both_low": []}
    for _, r in sub.iterrows():
        hi_loo, hi_ig = r["loo_imp"] >= xm, r["mean_abs_ig"] >= ym
        key = ("both_high" if hi_loo and hi_ig else "mech_only" if hi_ig else
               "loo_only" if hi_loo else "both_low")
        quad[key].append(r["gene"])

    colors = {"both_high": "#2ca02c", "mech_only": "#1f77b4",
              "loo_only": "#d62728", "both_low": "#999999"}
    label = {"both_high": "both-high (robust)", "mech_only": "mechanistic-only",
             "loo_only": "LOO-only", "both_low": "both-low"}

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.axvline(xm, color="k", lw=0.6, ls="--", alpha=0.5)
    ax.axhline(ym, color="k", lw=0.6, ls="--", alpha=0.5)
    for key, genes in quad.items():
        s = sub[sub["gene"].isin(genes)]
        ax.scatter(s["loo_imp"], s["mean_abs_ig"], s=40, c=colors[key], label=label[key])
    for _, r in sub.iterrows():
        ax.annotate(r["gene"], (r["loo_imp"], r["mean_abs_ig"]), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("LOO importance  −ΔAUC  (retrain-and-drop; larger = more important)")
    ax.set_ylabel("downstream-IG importance  mean|IG|  (fixed-model; larger = more important)")
    ax.set_title("LOO (causal) vs mechanistic gene importance")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    return quad


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def write_report(ctx, report_path: Path):
    df = ctx["df"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append(f"# Downstream-Classifier Gene Attribution — {pd.Timestamp.today():%Y-%m-%d}")
    lines.append("")
    lines.append(
        f"Attributes the RFS downstream classifier's decision back to the {ctx['n_genes']} "
        f"genes of `{ctx['meta']['gene_set']}` for run `{ctx['model_id']}`, by decomposing it "
        f"through the 128-dim shared embedding space. Complements the leave-one-out (LOO) ranking "
        f"(`reports/0629/0629_gene_importance.md`) and the whole-vector cosine-IG ranking "
        f"(`reports/0629/0629_gene_importance_ig.md`)."
    )
    lines.append("")
    if ctx["meta"].get("refit_gene_enc_from"):
        lines.append(
            f"**Model provenance.** `{ctx['model_id']}` is a refit "
            f"(`scripts/refit_gene_encoder.py`) of `{ctx['meta']['refit_gene_enc_from']}`: the image "
            f"encoder (and its cached embeddings) is reused **verbatim**, so the downstream LR is "
            f"identical to `{ctx['meta']['refit_gene_enc_from']}`'s (Soramic AUROC reproduced here = "
            f"**{ctx['soramic_auroc']:.3f}**), while only the GeneEncoder is re-aligned to that frozen "
            f"image space under a pinned, sorted gene order."
        )
        lines.append("")
    lines.append(
        "**Method.** (1) The headline downstream pipeline "
        "(`SimpleImputer → StandardScaler → SelectKBest(f_classif, k=100) → LR`, L1) is fit on the "
        "60 resection patients' image embeddings vs `rfs_2year`, then collapsed into a single decision "
        "direction `β ∈ R¹²⁸` with `logit(z) = β·z + b`. (2) The GeneEncoder Jacobian "
        "`J[d,j] = ∂geneenc(g)_d/∂g_j` (at the cohort-mean gene vector) gives each gene's effect on each "
        "shared dim; `C[d,j] = β_d·J[d,j]` is its contribution to the decision axis. (3) Integrated "
        f"Gradients of `s(g) = β·geneenc(g)` w.r.t. the gene input ({ctx['steps']} midpoint steps, "
        f"baseline = `{ctx['baseline']}`) aggregated over {ctx['n_patients']} patients as `mean|IG|` "
        "(importance) and `mean IG` (signed; **+ = pushes toward recurrence ≤ 2yr**)."
    )
    lines.append("")
    lines.append(
        "> **Caveat.** Genes never enter the predictor at inference — they shape the image encoder only "
        "through the contrastive alignment at training time. This is an *alignment-mediated proxy*, the "
        "per-dimension decomposition of the cosine target; it complements, not replaces, the LOO "
        "retrain-and-drop importance."
    )
    lines.append("")
    lines.append(f"- Completeness check (max |Σ_j IG − (s(x)−s(baseline))|): **{ctx['max_resid']:.2e}** "
                 f"= **{ctx['rel_resid']:.2e}** of the per-patient target range (s is unnormalized)")
    lines.append(f"- β reconstruction check (max |β·z+b − decision_function|): **{ctx['beta_recon']:.2e}**")
    lines.append(f"- Soramic AUROC of the unwound downstream LR: **{ctx['soramic_auroc']:.3f}** (target 0.732)")
    if ctx["rho_loo"] is not None:
        lines.append(f"- Spearman(downstream `mean|IG|`, LOO `−ΔAUC`): **rho = {ctx['rho_loo']:.3f}**, p = {ctx['p_loo']:.3f}")
    if ctx["rho_cos"] is not None:
        lines.append(f"- Spearman(downstream `mean|IG|`, cosine-IG `mean|IG|`): **rho = {ctx['rho_cos']:.3f}**, p = {ctx['p_cos']:.3f}")
    lines.append("")

    # Table 1: per-gene composite importance
    lines.append("## 1. Per-gene downstream importance (sorted by `mean|IG|`)")
    lines.append("")
    lines.append("| Gene | mean\\|IG\\| | signed mean IG | rank | LOO ΔAUC | LOO rank | cos-IG rank |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    ig_rank = df.sort_values("mean_abs_ig", ascending=False).reset_index(drop=True)
    loo_order = {g: i + 1 for i, g in enumerate(df.sort_values("loo_delta")["gene"])}
    cos_order = {g: i + 1 for i, g in enumerate(df.sort_values("cos_ig", ascending=False)["gene"])}
    for i, r in ig_rank.iterrows():
        loo_d = "—" if pd.isna(r["loo_delta"]) else f"{r['loo_delta']:+.3f}"
        loo_rk = "—" if pd.isna(r["loo_delta"]) else str(loo_order[r["gene"]])
        cos_rk = "—" if pd.isna(r.get("cos_ig")) else str(cos_order[r["gene"]])
        lines.append(
            f"| {r['gene']} | {r['mean_abs_ig']:.4f} | {r['signed_mean_ig']:+.4f} | "
            f"{i + 1} | {loo_d} | {loo_rk} | {cos_rk} |"
        )
    lines.append("")
    lines.append(f"![Downstream-IG vs cosine-IG]({ctx['scatter_name']})")
    lines.append("")

    # Table 2: top downstream dims
    lines.append("## 2. Top downstream dimensions by |β| (with bootstrap stability)")
    lines.append("")
    lines.append(f"Bootstrap = {ctx['n_boot']} stratified patient resamples. `sel. freq` = fraction of "
                 "resamples where SelectKBest+L1 keeps the dim. `top driver genes` = largest |C[d,j]|.")
    lines.append("")
    lines.append("| dim | β | bootstrap mean±sd | sel. freq | top driver genes (signed C) |")
    lines.append("|---|---:|---:|---:|---|")
    for d in ctx["top_dims"]:
        drivers = ctx["dim_drivers"][d]
        drv = ", ".join(f"{g} ({c:+.3f})" for g, c in drivers)
        lines.append(
            f"| {d} | {ctx['beta'][d]:+.3f} | {ctx['boot_mean'][d]:+.3f}±{ctx['boot_sd'][d]:.3f} | "
            f"{ctx['boot_freq'][d]:.2f} | {drv} |"
        )
    lines.append("")
    lines.append(f"![Gene → decision-axis contribution heatmap]({ctx['heatmap_name']})")
    lines.append("")

    # Section 3: LOO vs mechanistic reconciliation
    q = ctx["quad"]
    fmt = lambda gs: ", ".join(gs) if gs else "—"
    lines.append("## 3. LOO (causal) vs mechanistic importance")
    lines.append("")
    lines.append(
        "The two rankings answer different questions and largely disagree "
        f"(Spearman rho = {ctx['rho_loo']:.3f}, p = {ctx['p_loo']:.3f}; effectively unrelated). "
        "**LOO** (drop one gene, retrain the whole contrastive model, measure ΔAUC) is a *causal, "
        "end-to-end* importance that **lets correlated genes and the image encoder compensate** for the "
        "removed gene — and it measures the dominant channel by which genes act (shaping the image encoder "
        "at train time). **Mechanistic** importance (this report) holds the trained model and its image "
        "space **fixed** and asks how much each gene moves the decision axis — so it cannot see the "
        "image-encoder channel and credits genes the *current* model leans on, replaceable or not. "
        "Sampling noise (n=54, AUROC head-flips, unstable β) further weakens any correlation."
    )
    lines.append("")
    lines.append("Splitting genes at the median of each axis (see figure):")
    lines.append("")
    lines.append(f"- **both-high (robust)** — important by both, the most trustworthy: {fmt(q['both_high'])}")
    lines.append(f"- **mechanistic-only** — model leans on them but removal barely hurts ⇒ likely "
                 f"replaceable by correlated genes: {fmt(q['mech_only'])}")
    lines.append(f"- **LOO-only** — irreplaceable for performance but the fixed gene branch doesn't lean "
                 f"on them: {fmt(q['loo_only'])}")
    lines.append(f"- **both-low**: {fmt(q['both_low'])}")
    lines.append("")
    lines.append(f"![LOO vs mechanistic quadrants]({ctx['quadrant_name']})")
    lines.append("")
    lines.append("**Notes**")
    lines.append("")
    lines.append("- Stage-1 β says *which embedding dims the classifier uses*; stage-2 C and stage-3 IG say "
                 "*which genes move the patient along those dims*. A gene ranks high here only if it drives "
                 "dims the classifier weights — unlike cosine-IG, which weights all 128 dims equally.")
    lines.append("- Signed IG > 0 ⇒ higher expression pushes the patient toward 2-year recurrence along the "
                 "classifier's decision axis (orientation of `rfs_2year` positive class).")
    report_path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Stage 1: downstream decision direction
    pipe, beta_np, bias, X_res, y_res, embed_dim = fit_downstream(args.model_id)
    beta_recon = float(np.max(np.abs(X_res.values @ beta_np + bias - pipe.decision_function(X_res))))
    soramic_auroc = sanity_soramic_auroc(pipe, args.model_id)
    boot_mean, boot_sd, boot_freq = bootstrap_beta(X_res, y_res, embed_dim, args.n_boot)

    # Gene branch (reuses the IG loader; enforces deterministic gene order)
    gene_enc, gene_names, x, _z_img, meta, pids = load_inputs(args.model_id, device)
    beta = torch.tensor(beta_np, dtype=torch.float32, device=device)

    # Stage 2: Jacobian at the cohort-mean gene vector
    J = gene_jacobian(gene_enc, x.mean(0))  # (embed_dim, gene_dim)
    C = beta_np[:, None] * J  # (embed_dim, gene_dim)

    top_dims = np.argsort(-np.abs(beta_np))[: args.top_k]
    top_dims = [int(d) for d in top_dims if beta_np[d] != 0.0]
    dim_drivers = {}
    for d in top_dims:
        order = np.argsort(-np.abs(C[d]))[:3]
        dim_drivers[d] = [(gene_names[j], float(C[d, j])) for j in order]

    # Stage 3: IG on the decision target
    baseline = torch.zeros(x.shape[1], device=device) if args.baseline == "zero" else x.mean(0)
    ig, residual, delta_s, _ = integrated_gradients_beta(
        gene_enc, beta, x, baseline, args.steps, device
    )
    max_resid = float(residual.abs().max().cpu())
    # s(g) is unnormalized (unlike the bounded cosine target), so report the
    # completeness gap relative to the per-patient target range it must explain.
    rel_resid = float((residual.abs().max() / delta_s.abs().max().clamp_min(1e-12)).cpu())

    df = pd.DataFrame({
        "gene": gene_names,
        "mean_abs_ig": ig.abs().mean(0).cpu().numpy(),
        "signed_mean_ig": ig.mean(0).cpu().numpy(),
    })

    # Reference rankings
    loo = load_loo(args.loo_dir, args.baseline_auroc)
    df = df.merge(loo[["gene", "loo_delta"]], on="gene", how="left")
    if args.cos_ig_json.exists():
        cos = pd.DataFrame(json.loads(args.cos_ig_json.read_text())["genes"])[["gene", "mean_abs_ig"]]
        cos = cos.rename(columns={"mean_abs_ig": "cos_ig"})
        df = df.merge(cos, on="gene", how="left")
    else:
        df["cos_ig"] = np.nan

    def _spear(a, b):
        sub = df.dropna(subset=[a, b])
        if len(sub) < 3:
            return None, None
        return spearmanr(sub[a], sub[b])

    rho_loo, p_loo = _spear("mean_abs_ig", "loo_delta")
    if rho_loo is not None:
        rho_loo = -rho_loo  # compare to -ΔAUC (importance), keep sign convention
    rho_cos, p_cos = _spear("mean_abs_ig", "cos_ig")

    # Figures
    args.report.parent.mkdir(parents=True, exist_ok=True)
    heatmap_path = args.report.parent / (args.report.stem + "_heatmap.png")
    scatter_path = args.report.parent / (args.report.stem + "_scatter.png")
    quadrant_path = args.report.parent / (args.report.stem + "_quadrant.png")
    make_heatmap(C, gene_names, top_dims, heatmap_path)
    make_scatter(df, scatter_path)
    quad = make_quadrant(df, quadrant_path)

    # JSON
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "model_id": args.model_id,
        "gene_set": meta["gene_set"],
        "target": "beta . gene_enc(g)",
        "baseline": args.baseline,
        "steps": args.steps,
        "n_patients": len(pids),
        "n_genes": len(df),
        "select_k": SELECT_K,
        "n_boot": args.n_boot,
        "soramic_auroc": soramic_auroc,
        "completeness_max_residual": max_resid,
        "completeness_max_residual_relative": rel_resid,
        "beta_reconstruction_max_residual": beta_recon,
        "beta": beta_np.tolist(),
        "bias": bias,
        "bootstrap_beta_mean": boot_mean.tolist(),
        "bootstrap_beta_sd": boot_sd.tolist(),
        "bootstrap_selection_freq": boot_freq.tolist(),
        "top_dims": top_dims,
        "dim_drivers": {str(d): dim_drivers[d] for d in top_dims},
        "spearman_vs_loo": {"rho": rho_loo, "p": p_loo},
        "spearman_vs_cos_ig": {"rho": rho_cos, "p": p_cos},
        "loo_vs_mechanistic_quadrants": quad,
        "genes": df.sort_values("mean_abs_ig", ascending=False).to_dict(orient="records"),
    }, indent=2))
    print(f"Wrote results → {args.output}")

    ctx = dict(
        df=df, meta=meta, model_id=args.model_id, n_genes=len(df), n_patients=len(pids),
        steps=args.steps, baseline=args.baseline, n_boot=args.n_boot,
        soramic_auroc=soramic_auroc, max_resid=max_resid, beta_recon=beta_recon,
        beta=beta_np, boot_mean=boot_mean, boot_sd=boot_sd, boot_freq=boot_freq,
        top_dims=top_dims, dim_drivers=dim_drivers, rel_resid=rel_resid,
        rho_loo=rho_loo, p_loo=p_loo, rho_cos=rho_cos, p_cos=p_cos,
        heatmap_name=heatmap_path.name, scatter_name=scatter_path.name,
        quadrant_name=quadrant_path.name, quad=quad,
    )
    if args.report is not None:
        write_report(ctx, args.report)
        print(f"Wrote report → {args.report}")

    print(f"\nSoramic AUROC (downstream parity): {soramic_auroc:.3f} (target 0.732)")
    print(f"β reconstruction max residual: {beta_recon:.2e}")
    print(f"IG completeness max residual:  {max_resid:.2e} ({rel_resid:.2e} relative)")
    print("\nTop genes by downstream mean|IG|:")
    print(df.sort_values("mean_abs_ig", ascending=False)[
        ["gene", "mean_abs_ig", "signed_mean_ig", "loo_delta"]
    ].head(10).to_string(index=False))
    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-id", default="a0912600",
                   help="Refit contrastive run id (deterministic gene order required)")
    p.add_argument("--baseline", choices=["zero", "mean"], default="zero")
    p.add_argument("--steps", type=int, default=200, help="IG integration steps")
    p.add_argument("--n-boot", type=int, default=500, help="Bootstrap resamples for β stability")
    p.add_argument("--top-k", type=int, default=15, help="Top |β| dims to detail")
    p.add_argument("--loo-dir", type=Path, default=Path("results/eval/soramic/gene_ablation"))
    p.add_argument("--baseline-auroc", type=float, default=0.732)
    p.add_argument("--cos-ig-json", type=Path,
                   default=Path("results/eval/soramic/gene_ablation/integrated_gradients.json"),
                   help="Cosine-IG JSON for cross-comparison (optional)")
    p.add_argument("--output", type=Path,
                   default=Path("results/eval/soramic/gene_ablation/downstream_gene_attribution.json"))
    p.add_argument("--report", type=Path,
                   default=Path("reports/0629/0629_downstream_gene_attribution.md"))
    args = p.parse_args()
    if args.report is not None and str(args.report) in ("", "."):
        args.report = None
    return args


if __name__ == "__main__":
    run(parse_args())
