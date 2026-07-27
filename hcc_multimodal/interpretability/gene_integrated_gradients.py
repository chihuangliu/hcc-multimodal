"""Integrated-Gradients gene importance for a contrastive image-gene model.

Computes per-gene Integrated Gradients (IG) attributions on the cross-modal
alignment target

    F_i(g) = cos( z_img_i , gene_enc(g) )

where ``z_img_i`` is patient *i*'s cached (frozen) mean-pooled image embedding
and ``g`` is the patient's gene-expression vector. IG is integrated only through
the ``GeneEncoder`` w.r.t. the gene input (no ViT backprop). Per-patient
attributions are aggregated into a model-level ranking via ``mean|IG|`` across
the resection cohort, then compared to the leave-one-out (LOO) gene-ablation
Delta-AUC ranking (reports/0629) with Spearman correlation.

Why the alignment target: at inference the downstream RFS head consumes image
embeddings only -- genes never enter the predictor's forward path. Genes act on
the model solely by shaping the image encoder through the contrastive alignment
at training time. So IG here measures alignment sensitivity: a cheap, single-
model proxy for -- not a replacement of -- the LOO retrain-and-drop importance.

Example:
    python scripts/gene_integrated_gradients.py \
        --model-id 9109a6c2 \
        --baseline zero \
        --steps 50 \
        --loo-dir results/eval/soramic/gene_ablation \
        --baseline-auroc 0.732
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
import torch.nn.functional as F
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hcc_multimodal.contrastive.data import _load_gene_matrix
from hcc_multimodal.contrastive.encoders import GeneEncoder
from hcc_multimodal.contrastive.train import _GENE_SETS

TRAINING_ROOT = Path("training/contrastive")
EMB_CACHE = "cached_embeddings/resection_img_emb.parquet"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_inputs(model_id: str, device: torch.device):
    """Load the trained GeneEncoder, the CPM gene matrix, and cached image
    embeddings, aligned on the shared set of resection patient ids."""
    run_dir = TRAINING_ROOT / model_id
    meta = json.loads((run_dir / "metadata.json").read_text())

    if not meta.get("deterministic_gene_order", False):
        raise ValueError(
            f"Run '{model_id}' was trained with a non-deterministic gene column order "
            f"(genes passed as a set), so its GeneEncoder's gene->slot mapping is "
            f"unrecoverable and IG attributions would be scrambled. Use a model produced "
            f"by scripts/refit_gene_encoder.py (which pins a sorted order), e.g. the refit "
            f"of this run."
        )

    genes = set(_GENE_SETS[meta["gene_set"]])
    # Reproduce the run's own column order. Runs predating the --sort_genes flag
    # always sorted, so True is the right fallback despite the flag now
    # defaulting to False. The equality check below is the backstop.
    gene_df = _load_gene_matrix(genes, sort_genes=meta.get("sort_genes", True))
    gene_names = gene_df.columns.tolist()
    if "gene_order" in meta and list(meta["gene_order"]) != gene_names:
        raise ValueError(
            "Loaded gene column order does not match the order stored in the model's "
            f"metadata.\n  metadata: {list(meta['gene_order'])}\n  loaded:   {gene_names}"
        )

    img_df = pd.read_parquet(run_dir / EMB_CACHE)  # (patients x 128), int pid index

    pids = sorted(set(gene_df.index) & set(img_df.index))
    gene_df = gene_df.loc[pids]
    img_df = img_df.loc[pids]

    gene_dim = gene_df.shape[1]
    gene_enc = GeneEncoder(gene_dim, meta["gene_hidden_dim"], meta["embed_dim"]).to(device)
    ckpt = torch.load(run_dir / "best_model.pt", map_location=device, weights_only=False)
    in_features = ckpt["gene_enc"]["net.0.weight"].shape[1]
    if in_features != gene_dim:
        raise ValueError(
            f"Checkpoint GeneEncoder expects {in_features} genes but loaded matrix has "
            f"{gene_dim} ({gene_names}). Gene-set mismatch."
        )
    gene_enc.load_state_dict(ckpt["gene_enc"])
    gene_enc.eval()

    x = torch.tensor(gene_df.values, dtype=torch.float32, device=device)       # (P, G)
    z_img = torch.tensor(img_df.values, dtype=torch.float32, device=device)    # (P, D)
    return gene_enc, gene_names, x, z_img, meta, pids


# ---------------------------------------------------------------------------
# Integrated Gradients
# ---------------------------------------------------------------------------
def _alignment(gene_enc, z_img_n, g):
    """Per-patient cosine similarity cos(z_img, gene_enc(g)). z_img_n is
    pre-normalized; returns (P,)."""
    z_gene_n = F.normalize(gene_enc(g), dim=-1)
    return (z_img_n * z_gene_n).sum(dim=-1)


def integrated_gradients(gene_enc, z_img, x, baseline, steps, device):
    """Midpoint-rule Integrated Gradients of the alignment target w.r.t. the
    gene input. Patients are independent rows, so a single batched backward per
    step yields correct per-patient gradients.

    Returns (ig, residual, delta_F) where ig is (P, G), residual is the
    completeness gap sum_j ig - (F(x) - F(baseline)) per patient (should be ~0),
    and delta_F = F(x) - F(baseline) per patient.
    """
    z_img_n = F.normalize(z_img, dim=-1)
    P, G = x.shape
    base = baseline.unsqueeze(0).expand(P, G)
    total_grad = torch.zeros_like(x)

    # midpoint alphas: (k - 0.5) / steps for k = 1..steps -> best completeness
    alphas = (torch.arange(steps, device=device).float() + 0.5) / steps
    for a in alphas:
        g = (base + a * (x - base)).clone().requires_grad_(True)
        sim = _alignment(gene_enc, z_img_n, g)
        sim.sum().backward()
        total_grad += g.grad

    avg_grad = total_grad / steps
    ig = (x - base) * avg_grad

    with torch.no_grad():
        f_x = _alignment(gene_enc, z_img_n, x)
        f_base = _alignment(gene_enc, z_img_n, base)
    delta_f = f_x - f_base
    residual = ig.sum(dim=-1) - delta_f
    return ig.detach(), residual.detach(), delta_f.detach()


# ---------------------------------------------------------------------------
# LOO reference
# ---------------------------------------------------------------------------
def load_loo(loo_dir: Path, baseline_auroc: float) -> pd.DataFrame:
    """Parse leave-one-out gene-ablation eval JSONs into a per-gene Delta-AUC
    table. Filename form: embedding_<GENE>_<RUNID>.json (gene names contain dots,
    so split the run id off the right)."""
    rows = []
    for p in sorted(loo_dir.glob("embedding_*.json")):
        body = p.stem[len("embedding_"):]
        gene, _runid = body.rsplit("_", 1)
        d = json.loads(p.read_text())
        emb = d["results"]["average"]["embedding"]
        auroc = max(emb["lr"]["auroc"], emb["rf"]["auroc"])
        rows.append({"gene": gene, "loo_auroc": auroc, "loo_delta": auroc - baseline_auroc})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def write_report(df: pd.DataFrame, spear, meta, args, max_resid, n_patients, fig_path: Path, report_path: Path):
    rho, pval = spear
    report_path.parent.mkdir(parents=True, exist_ok=True)
    ig_rank = df.sort_values("mean_abs_ig", ascending=False).reset_index(drop=True)
    loo_rank = df.sort_values("loo_delta").reset_index(drop=True)  # most negative first

    lines = []
    lines.append(f"# Gene Importance via Integrated Gradients — {pd.Timestamp.today():%Y-%m-%d}")
    lines.append("")
    lines.append(
        f"Integrated-Gradients (IG) gene importance for run `{args.model_id}` "
        f"(gene set `{meta['gene_set']}`, {len(df)} genes), compared to the "
        f"leave-one-out (LOO) ranking in `reports/0629/0629_gene_importance.md`."
    )
    lines.append("")
    if meta.get("refit_gene_enc_from"):
        lines.append(
            f"**Model provenance.** The original run `{meta['refit_gene_enc_from']}` stored its "
            f"genes in a non-deterministic column order (genes passed as a Python `set`), which "
            f"was never persisted — so its trained GeneEncoder's gene→slot mapping is "
            f"unrecoverable. `{args.model_id}` is a refit (`scripts/refit_gene_encoder.py`): the "
            f"image encoder of `{meta['refit_gene_enc_from']}` is reused **verbatim** (so "
            f"downstream RFS performance is identical by construction — Soramic best-head AUROC "
            f"= {args.baseline_auroc:.3f}) while only the GeneEncoder is re-trained against that "
            f"frozen image space using a pinned, sorted gene order. IG below is therefore "
            f"computed on a gene branch re-aligned to `{meta['refit_gene_enc_from']}`'s image "
            f"representation, not the original co-trained one."
        )
        lines.append("")
    lines.append("**Method.** At inference the downstream RFS head consumes image embeddings only — "
                 "genes never enter the predictor. Genes act on the model solely by shaping the image "
                 "encoder through the contrastive alignment during training. IG is therefore computed on "
                 "the cross-modal alignment target")
    lines.append("")
    lines.append("    F_i(g) = cos( z_img_i , gene_enc(g) )")
    lines.append("")
    lines.append(
        f"with patient *i*'s cached mean-pooled image embedding `z_img_i` held fixed; IG is integrated "
        f"only through the GeneEncoder w.r.t. the gene input ({args.steps} midpoint steps, baseline = "
        f"`{args.baseline}`). Per-patient attributions over {n_patients} resection patients are "
        f"aggregated as `mean|IG|`. **IG measures alignment sensitivity — a proxy for, not a replacement "
        f"of, the LOO retrain-and-drop importance.**"
    )
    lines.append("")
    lines.append(f"- Completeness check (max |sum_j IG − (F(x)−F(baseline))| over patients): **{max_resid:.2e}**")
    lines.append(f"- Spearman( IG importance `mean|IG|`, LOO importance `−ΔAUC` ): "
                 f"**rho = {rho:.3f}**, p = {pval:.3f} (positive ⇒ agreement)")
    lines.append("")

    # Per-gene table sorted by IG importance
    lines.append("## Per-gene importance (sorted by `mean|IG|`)")
    lines.append("")
    lines.append("| Gene | mean\\|IG\\| | signed mean IG | IG rank | LOO ΔAUC | LOO rank |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    ig_order = {g: i + 1 for i, g in enumerate(ig_rank["gene"])}
    loo_order = {g: i + 1 for i, g in enumerate(loo_rank["gene"])}
    for _, r in ig_rank.iterrows():
        loo_d = "—" if pd.isna(r["loo_delta"]) else f"{r['loo_delta']:+.3f}"
        loo_rk = "—" if pd.isna(r["loo_delta"]) else str(loo_order[r["gene"]])
        lines.append(
            f"| {r['gene']} | {r['mean_abs_ig']:.4f} | {r['signed_mean_ig']:+.4f} | "
            f"{ig_order[r['gene']]} | {loo_d} | {loo_rk} |"
        )
    lines.append("")
    lines.append(f"![IG vs LOO]({fig_path.name})")
    lines.append("")
    lines.append("**Notes**")
    lines.append("")
    lines.append("- IG ranks genes by how much the *current* model's image–gene alignment depends on each "
                 "gene; LOO ranks by how much downstream AUC drops when the gene is removed and the model "
                 "retrained. They answer different questions, so disagreement is expected and informative "
                 "(e.g. a high-IG gene with small LOO effect is likely replaceable by a correlated gene).")
    report_path.write_text("\n".join(lines) + "\n")


def make_figure(df: pd.DataFrame, fig_path: Path):
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    sub = df.dropna(subset=["loo_delta"])
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(sub["loo_delta"], sub["mean_abs_ig"], s=30)
    for _, r in sub.iterrows():
        ax.annotate(r["gene"], (r["loo_delta"], r["mean_abs_ig"]), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("LOO ΔAUC  (more negative = more important)")
    ax.set_ylabel("IG importance  mean|IG|  (larger = more important)")
    ax.set_title("Integrated Gradients vs leave-one-out gene importance")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(args) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gene_enc, gene_names, x, z_img, meta, pids = load_inputs(args.model_id, device)

    if args.baseline == "zero":
        baseline = torch.zeros(x.shape[1], device=device)
    elif args.baseline == "mean":
        baseline = x.mean(dim=0)
    else:
        raise ValueError(args.baseline)

    ig, residual, _delta_f = integrated_gradients(
        gene_enc, z_img, x, baseline, args.steps, device
    )
    max_resid = float(residual.abs().max().cpu())

    mean_abs = ig.abs().mean(dim=0).cpu().numpy()
    signed_mean = ig.mean(dim=0).cpu().numpy()
    df = pd.DataFrame({
        "gene": gene_names,
        "mean_abs_ig": mean_abs,
        "signed_mean_ig": signed_mean,
    })

    loo = load_loo(args.loo_dir, args.baseline_auroc)
    df = df.merge(loo, on="gene", how="left")

    paired = df.dropna(subset=["loo_delta"])
    if len(paired) >= 3:
        rho, pval = spearmanr(paired["mean_abs_ig"], -paired["loo_delta"])
    else:
        rho, pval = float("nan"), float("nan")

    df = df.sort_values("mean_abs_ig", ascending=False).reset_index(drop=True)

    # Outputs
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "model_id": args.model_id,
        "gene_set": meta["gene_set"],
        "target": "cosine(z_img, gene_enc(g))",
        "baseline": args.baseline,
        "steps": args.steps,
        "n_patients": len(pids),
        "n_genes": len(df),
        "baseline_auroc": args.baseline_auroc,
        "completeness_max_residual": max_resid,
        "spearman_ig_vs_neg_loo_delta": {"rho": None if np.isnan(rho) else rho,
                                         "p": None if np.isnan(pval) else pval},
        "genes": df.to_dict(orient="records"),
    }
    args.output.write_text(json.dumps(out, indent=2))
    print(f"Wrote results → {args.output}")

    if args.report is not None:
        fig_path = args.report.with_suffix("").parent / (args.report.stem + "_scatter.png")
        make_figure(df, fig_path)
        write_report(df, (rho, pval), meta, args, max_resid, len(pids), fig_path, args.report)
        print(f"Wrote report → {args.report}")
        print(f"Wrote figure → {fig_path}")

    print(f"\nCompleteness max residual: {max_resid:.2e}")
    print(f"Spearman(mean|IG|, -LOO ΔAUC) = {rho:.3f} (p={pval:.3f}); positive = agreement")
    print("\nTop genes by mean|IG|:")
    print(df[["gene", "mean_abs_ig", "loo_delta"]].head(10).to_string(index=False))
    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-id", default="a0912600",
                   help="Contrastive run id (must have a deterministic gene order, "
                        "i.e. a refit_gene_encoder.py output)")
    p.add_argument("--baseline", choices=["zero", "mean"], default="zero",
                   help="IG baseline: zero vector or cohort-mean gene vector")
    p.add_argument("--steps", type=int, default=200, help="IG integration steps (midpoint rule)")
    p.add_argument("--loo-dir", type=Path,
                   default=Path("results/eval/soramic/gene_ablation"),
                   help="Dir of leave-one-out embedding_<GENE>_<RUNID>.json files")
    p.add_argument("--baseline-auroc", type=float, default=0.732,
                   help="Full-gene-set baseline AUROC for LOO ΔAUC (run 9109a6c2 = 0.732)")
    p.add_argument("--output", type=Path,
                   default=Path("results/eval/soramic/gene_ablation/integrated_gradients.json"),
                   help="Output JSON path")
    p.add_argument("--report", type=Path,
                   default=Path("reports/0629/0629_gene_importance_ig.md"),
                   help="Markdown report path (set to '' to skip)")
    args = p.parse_args()
    if args.report is not None and str(args.report) in ("", "."):
        args.report = None
    return args


if __name__ == "__main__":
    run(parse_args())
