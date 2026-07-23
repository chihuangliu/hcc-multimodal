"""PCA projection of cohorts onto training-fit axes (report §5).

For each model_id, fits PCA on the fit-cohort (default resection) and projects all
cohorts into the same 2 components — one subplot per model. Disjoint target points
(a6f970d6) vs overlapping (dc7e1d10) is the label-free tell.

Usage:
  python -m hcc_multimodal.eval.diagnose.pca a6f970d6 dc7e1d10 \
      --out reports/0720/drift_pca_a6_vs_dc.png
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from hcc_multimodal.eval.diagnose.common import COHORTS, load_embeddings  # noqa: E402

_COLOR = {"resection": "#2166ac", "soramic": "#b2182b", "lausanne": "#f4a582"}


def _plot_model(ax, model_id: str, fit_cohort: str, cohorts: list[str], labeled: bool = False) -> None:
    embs = {c: load_embeddings(model_id, c, labeled=labeled) for c in cohorts}
    pca = PCA(2).fit(embs[fit_cohort])
    for c in cohorts:
        z = pca.transform(embs[c])
        ax.scatter(z[:, 0], z[:, 1], s=22, alpha=0.7, c=_COLOR.get(c), edgecolors="none", label=c)
    ev = pca.explained_variance_ratio_ * 100
    ax.set_title(model_id, fontsize=10)
    ax.set_xlabel(f"PC1 ({ev[0]:.0f}%)")
    ax.set_ylabel(f"PC2 ({ev[1]:.0f}%)")
    ax.legend(fontsize=8, loc="best")


def run(model_ids: list[str], fit_cohort: str, cohorts: list[str], out: Path, labeled: bool = False) -> Path:
    fig, axes = plt.subplots(1, len(model_ids), figsize=(6 * len(model_ids), 5.2), squeeze=False)
    for ax, mid in zip(axes[0], model_ids):
        _plot_model(ax, mid, fit_cohort, cohorts, labeled)
    fig.suptitle(f"Image embeddings projected onto {fit_cohort}-fit PCA (label-free)", fontsize=12)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved → {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_ids", nargs="+", help="one or more model_ids (one subplot each)")
    ap.add_argument("--fit-cohort", default="resection", choices=COHORTS, help="cohort the PCA is fit on")
    ap.add_argument("--cohort", nargs="+", default=list(COHORTS), choices=COHORTS, help="cohorts to plot")
    ap.add_argument("--labeled-only", action="store_true", help="restrict to 2yr-RFS-labeled SIDs")
    ap.add_argument("--out", type=Path, default=Path("reports/0720/drift_pca.png"))
    args = ap.parse_args()
    run(args.model_ids, args.fit_cohort, args.cohort, args.out, args.labeled_only)


if __name__ == "__main__":
    main()
