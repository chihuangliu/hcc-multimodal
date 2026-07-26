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

Must be run on a refit model with a deterministic gene order (e.g. ``92ae6a23``,
the refit of ``dc7e1d10``): the image encoder is reused verbatim, so the downstream
head is identical to the source model's by construction.
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
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.utils import resample

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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


def transfer_auroc(pipe, model_id: str, cohort: str) -> float:
    """Resection-fit head → external-cohort AUROC (uses positive_scores so Ridge's
    proba-less decision_function is handled)."""
    X = load_embeddings(model_id, cohort)
    y = load_survival_outcomes(cohort)["rfs_2year"].dropna().astype(int)
    common = X.index.intersection(y.index)
    scores = positive_scores(pipe, X.loc[common])
    return compute_metrics(y.loc[common].values, scores)["auroc"]


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
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
    L.append(
        f"**Model provenance.** `{ctx['model_id']}` is a refit "
        f"(`scripts`/`hcc_multimodal.interpretability.refit_gene_encoder`, "
        f"`--precompute-embeddings`) of `{ctx['refit_from']}`: the image encoder and its "
        f"cached embeddings are reused **verbatim** (resection embedding cache is "
        f"byte-identical), so the downstream head is identical to `{ctx['refit_from']}`'s "
        f"by construction — its Setting-A grid and survival results (incl. head A1) carry "
        f"over unchanged. Only the GeneEncoder is re-aligned to that frozen image space "
        f"under a pinned, sorted gene order, giving an interpretable gene branch."
    )
    L.append("")
    L.append(
        f"**Method.** (1) The grid head (`SimpleImputer(median) → StandardScaler → "
        f"{ctx['fs']}(k={ctx['select_k']}) → {ctx['model']}`{ctx['params_str']}) is fit on the "
        f"{ctx['n_res']} resection patients' image embeddings vs `rfs_2year`, then collapsed "
        f"into a single decision direction `β ∈ R¹²⁸` with `logit(z) = β·z + b`. (2) The "
        f"GeneEncoder Jacobian `J[d,j] = ∂geneenc(g)_d/∂g_j` (at the cohort-mean gene vector) "
        f"gives each gene's effect on each shared dim; `C[d,j] = β_d·J[d,j]` is its "
        f"contribution to the decision axis. (3) Integrated Gradients of `s(g) = β·geneenc(g)` "
        f"w.r.t. the gene input ({ctx['steps']} midpoint steps, baseline = `{ctx['baseline']}`) "
        f"aggregated over {ctx['n_patients']} patients as `mean|IG|` (importance) and "
        f"`mean IG` (signed; **+ = pushes toward recurrence ≤ 2yr**)."
    )
    L.append("")
    L.append(
        "> **Caveat.** Genes never enter the predictor at inference — they shape the image "
        "encoder only through the contrastive alignment at training time. This is an "
        "*alignment-mediated proxy*: the per-dimension decomposition of the decision axis "
        "on a re-aligned gene branch."
    )
    L.append("")
    L.append(f"- Completeness check (max |Σ_j IG − (s(x)−s(baseline))|): **{ctx['max_resid']:.2e}** "
             f"= **{ctx['rel_resid']:.2e}** of the per-patient target range")
    L.append(f"- β reconstruction check (max |β·z+b − decision_function|): **{ctx['beta_recon']:.2e}**")
    L.append(f"- Resection 3-fold CV AUROC of the head: **{ctx['cv_auc']:.3f}** "
             f"(dc7e1d10 grid best cell = 0.744)")
    L.append(f"- Soramic transfer AUROC: **{ctx['soramic']:.3f}** (dc7e1d10 best cell = 0.709); "
             f"Lausanne: **{ctx['lusanne']:.3f}**")
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
    L.append("## 2. Top downstream dimensions by |β| (with bootstrap stability)")
    L.append("")
    L.append(f"Bootstrap = {ctx['n_boot']} stratified patient resamples. `sel. freq` = fraction of "
             "resamples where the selector keeps the dim. `top driver genes` = largest |C[d,j]|.")
    L.append("")
    L.append("| dim | β | bootstrap mean±sd | sel. freq | top driver genes (signed C) |")
    L.append("|---|---:|---:|---:|---|")
    for d in ctx["top_dims"]:
        drv = ", ".join(f"{g} ({c:+.3f})" for g, c in ctx["dim_drivers"][d])
        L.append(f"| {d} | {ctx['beta'][d]:+.3f} | {ctx['boot_mean'][d]:+.3f}±{ctx['boot_sd'][d]:.3f} | "
                 f"{ctx['boot_freq'][d]:.2f} | {drv} |")
    L.append("")
    L.append("**Notes**")
    L.append("")
    L.append("- Stage-1 β says *which embedding dims the classifier uses*; stage-2 C and stage-3 IG "
             "say *which genes move the patient along those dims*. A gene ranks high only if it drives "
             "dims the classifier weights.")
    L.append("- Signed IG > 0 ⇒ higher expression pushes the patient toward 2-year recurrence along "
             "the classifier's decision axis.")
    report_path.write_text("\n".join(L) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_params = json.loads(args.model_params) if args.model_params else {}

    # Stage 1
    pipe, beta_np, bias, X_res, y_res, embed_dim = fit_downstream(
        args.model_id, args.fs, args.model, args.select_k, model_params
    )
    beta_recon = float(np.max(np.abs(
        X_res.values @ beta_np + bias - pipe.decision_function(X_res)
    )))
    cv_auc = resection_cv_auc(X_res, y_res, args.fs, args.model, args.select_k, model_params)
    soramic = transfer_auroc(pipe, args.model_id, "soramic")
    lusanne = transfer_auroc(pipe, args.model_id, "lusanne")
    boot_mean, boot_sd, boot_freq = bootstrap_beta(
        X_res, y_res, embed_dim, args.fs, args.model, args.select_k, model_params, args.n_boot
    )

    # Gene branch
    gene_enc, gene_names, x, _z_img, meta, pids = load_inputs(args.model_id, device)
    beta = torch.tensor(beta_np, dtype=torch.float32, device=device)

    # Stage 2: Jacobian
    J = gene_jacobian(gene_enc, x.mean(0))
    C = beta_np[:, None] * J
    top_dims = [int(d) for d in np.argsort(-np.abs(beta_np))[: args.top_k] if beta_np[d] != 0.0]
    dim_drivers = {}
    for d in top_dims:
        order = np.argsort(-np.abs(C[d]))[:3]
        dim_drivers[d] = [(gene_names[j], float(C[d, j])) for j in order]

    # Stage 3: IG
    baseline = torch.zeros(x.shape[1], device=device) if args.baseline == "zero" else x.mean(0)
    ig, residual, delta_s = integrated_gradients_beta(gene_enc, beta, x, baseline, args.steps, device)
    max_resid = float(residual.abs().max().cpu())
    rel_resid = float((residual.abs().max() / delta_s.abs().max().clamp_min(1e-12)).cpu())

    df = pd.DataFrame({
        "gene": gene_names,
        "mean_abs_ig": ig.abs().mean(0).cpu().numpy(),
        "signed_mean_ig": ig.mean(0).cpu().numpy(),
    })

    params_str = f", α={model_params['model__alpha']}" if "model__alpha" in model_params else ""
    head = f"{args.model}/{args.fs} k={args.select_k}"

    # Figure
    args.report.parent.mkdir(parents=True, exist_ok=True)
    heatmap_path = args.report.parent / (args.report.stem + "_heatmap.png")
    make_heatmap(C, gene_names, top_dims, heatmap_path)

    # JSON
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "model_id": args.model_id, "refit_from": meta.get("refit_gene_enc_from"),
        "gene_set": meta["gene_set"], "head": head, "model_params": model_params,
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
        max_resid=max_resid, rel_resid=rel_resid, beta_recon=beta_recon,
        beta=beta_np, boot_mean=boot_mean, boot_sd=boot_sd, boot_freq=boot_freq,
        top_dims=top_dims, dim_drivers=dim_drivers, heatmap_name=heatmap_path.name,
    )
    write_report(ctx, args.report)
    print(f"Wrote report → {args.report}")

    print(f"\nHead: {head}{params_str}")
    print(f"Resection CV AUROC: {cv_auc:.3f} | Soramic: {soramic:.3f} | Lausanne: {lusanne:.3f}")
    print(f"β reconstruction max residual: {beta_recon:.2e}")
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
