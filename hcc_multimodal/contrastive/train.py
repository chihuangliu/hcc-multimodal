"""Contrastive MRI-gene training script."""

import argparse
import csv
import json
import subprocess
import uuid
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import v2
from tqdm import tqdm

import hcc_multimodal
from hcc_multimodal.contrastive.data import build_dataset
from hcc_multimodal.contrastive.encoders import BACKBONES, BACKBONE_TRANSFORMS, GeneEncoder, ImageEncoder
from hcc_multimodal.contrastive.config import (
    GENE_SET,
    PREDEFINED_HCC_2Y_CV_GENES,
    RNA_2Y_BEFORE_CV_GENES,
)
from hcc_multimodal.contrastive.loss import contrastive_loss

_GENE_SETS = {
    "all": GENE_SET,
    "predefined_2y_cv": PREDEFINED_HCC_2Y_CV_GENES,
    "2y_before_cv": RNA_2Y_BEFORE_CV_GENES,
}

_OUT_ROOT = Path(hcc_multimodal.__file__).parent.parent / "training" / "contrastive"


def _setup_run(args: argparse.Namespace) -> Path:
    run_id = uuid.uuid4().hex[:8]
    run_dir = _OUT_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        git_hash = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        git_hash = "unknown"
    meta = {**vars(args), "run_id": run_id, "git_commit": git_hash}
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    return run_dir


def _record_gene_order(run_dir: Path, gene_order: list[str]) -> None:
    """Merge the resolved gene column order into an existing metadata.json.

    Written as a second pass once the dataset is built: `_setup_run` writes
    metadata.json up front so a run that dies during dataset construction still
    leaves a readable file for the eval tooling that scans run directories.
    A gene's GeneEncoder input-slot index is its position in `gene_order`.
    """
    meta_path = run_dir / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["gene_order"] = gene_order
    meta["deterministic_gene_order"] = True
    meta_path.write_text(json.dumps(meta, indent=2))


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    run_dir = _setup_run(args)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    augment = v2.Compose(
        [
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            BACKBONE_TRANSFORMS[args.model](),
        ]
    )

    genes = set(_GENE_SETS[args.gene_set])
    if args.drop_genes:
        missing = set(args.drop_genes) - genes
        if missing:
            raise ValueError(
                f"--drop-genes contains genes not in gene set '{args.gene_set}': {sorted(missing)}"
            )
        genes = genes - set(args.drop_genes)
        print(f"Dropped {len(args.drop_genes)} gene(s): {sorted(args.drop_genes)} "
              f"({len(genes)} remaining)")

    dataset = build_dataset(
        n_per_axis=args.n_per_axis,
        axes=args.axes or None,
        outcome_col=args.outcome_col,
        img_size=args.img_size,
        transform=augment,
        mri_type=args.mri_type,
        genes=genes,
        sort_genes=args.sort_genes,
        bbox_pad=args.bbox_pad,
    )

    gene_order = list(dataset.gene_matrix.columns)
    _record_gene_order(run_dir, gene_order)
    print(f"Gene order ({len(gene_order)}, sort_genes={args.sort_genes}): {gene_order}")

    if args.split_unit == "patient":
        patients = sorted({pid for pid, _, _ in dataset._index})
        patient_labels = [int(dataset.outcomes[pid]) for pid in patients]
        train_pids, val_pids = train_test_split(
            patients, test_size=args.val_split, stratify=patient_labels, random_state=args.seed
        )
        train_pid_set, val_pid_set = set(train_pids), set(val_pids)
        train_idx = [i for i, (pid, _, _) in enumerate(dataset._index) if pid in train_pid_set]
        val_idx = [i for i, (pid, _, _) in enumerate(dataset._index) if pid in val_pid_set]
    else:
        labels = [int(dataset.outcomes[pid]) for pid, _, _ in dataset._index]
        indices = list(range(len(dataset)))
        train_idx, val_idx = train_test_split(
            indices, test_size=args.val_split, stratify=labels, random_state=args.seed
        )
    train_ds, val_ds = Subset(dataset, train_idx), Subset(dataset, val_idx)

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
    )

    gene_dim = dataset.gene_matrix.shape[1]
    img_enc = ImageEncoder(args.model, args.embed_dim, args.freeze_backbone).to(device)
    gene_enc = GeneEncoder(gene_dim, args.gene_hidden_dim, args.embed_dim).to(device)

    if args.base_model is not None:
        ckpt_path = _OUT_ROOT / args.base_model / "best_model.pt"
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        img_enc.load_state_dict(ckpt["img_enc"])
        gene_enc.load_state_dict(ckpt["gene_enc"])
        print(f"Loaded weights from {ckpt_path}")

    optimizer = torch.optim.AdamW(
        [p for p in list(img_enc.parameters()) + list(gene_enc.parameters()) if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    csv_path = run_dir / "losses.csv"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss"])

    best_val = float("inf")
    step = 0
    for epoch in range(1, args.epochs + 1):
        img_enc.train()
        gene_enc.train()
        total_train = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{args.epochs} train", leave=False)
        for imgs, genes, outcomes, _ in train_pbar:
            imgs, genes, outcomes = (
                imgs.to(device),
                genes.to(device),
                outcomes.to(device),
            )
            loss = contrastive_loss(
                img_enc(imgs),
                gene_enc(genes),
                outcomes=outcomes,
                temperature=args.temperature,
                lam=args.lam,
                step=step,
                reg_mode=args.reg_mode,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train += loss.item()
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")
            step += 1
        scheduler.step()

        img_enc.eval()
        gene_enc.eval()
        total_val = 0.0
        with torch.no_grad():
            for imgs, genes, outcomes, _ in tqdm(val_loader, desc=f"Epoch {epoch:3d}/{args.epochs} val  ", leave=False):
                imgs, genes, outcomes = (
                    imgs.to(device),
                    genes.to(device),
                    outcomes.to(device),
                )
                total_val += contrastive_loss(
                    img_enc(imgs),
                    gene_enc(genes),
                    outcomes=outcomes,
                    temperature=args.temperature,
                    lam=args.lam,
                    step=step,
                    reg_mode=args.reg_mode,
                ).item()

        avg_train = total_train / len(train_loader)
        avg_val = total_val / max(1, len(val_loader))
        print(
            f"Epoch {epoch:3d}/{args.epochs}  train={avg_train:.4f}  val={avg_val:.4f}"
        )

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, avg_train, avg_val])

        if avg_val < best_val:
            best_val = avg_val
            torch.save(
                {"img_enc": img_enc.state_dict(), "gene_enc": gene_enc.state_dict()},
                run_dir / "best_model.pt",
            )

    torch.save(
        {"img_enc": img_enc.state_dict(), "gene_enc": gene_enc.state_dict()},
        run_dir / "last_model.pt",
    )
    print(f"Done. Best val loss: {best_val:.4f}  →  {run_dir}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train contrastive MRI-gene model")

    p.add_argument("--model", default="vit_b_32", choices=list(BACKBONES))
    p.add_argument(
        "--base_model", default=None, metavar="RUN_ID",
        help="Run ID to continue training from (loads best_model.pt weights before epoch 1).",
    )
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--gene_hidden_dim", type=int, default=256)
    p.add_argument("--freeze_backbone", action="store_true")

    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--lam", type=float, default=0.1)
    p.add_argument(
        "--reg_mode", default="per_modality", choices=["per_modality", "average"]
    )

    p.add_argument(
        "--gene_set",
        default="all",
        choices=list(_GENE_SETS),
        help="Gene set to use. 'all'=full RNA-seq; 'predefined_2y_cv'=PREDEFINED_HCC_2Y_CV_GENES; '2y_before_cv'=RNA_2Y_BEFORE_CV_GENES.",
    )
    p.add_argument(
        "--drop-genes",
        nargs="+",
        default=[],
        metavar="GENE",
        help="Gene symbol(s) to drop from the selected gene set (ablation). "
        "Each must be present in the gene set.",
    )
    p.add_argument(
        "--sort_genes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Order gene columns alphabetically. Default (--no-sort_genes) keeps the "
        "RNA-seq CSV's row order. Either way the resolved order is recorded as "
        "'gene_order' in the run's metadata.json.",
    )
    p.add_argument(
        "--n_per_axis",
        type=lambda v: None if v == "all" else int(v),
        default=None,
        metavar="N|all",
        help="Slices per axis per patient. 'all' uses every slice. Default: all.",
    )
    p.add_argument(
        "--axes",
        type=int,
        nargs="+",
        default=0,
        metavar="AXIS",
        help="Axes to slice (0=sagittal 1=coronal 2=axial). Default: 0.",
    )
    p.add_argument(
        "--outcome_col", default="rfs_2year", choices=["rfs_1year", "rfs_2year"]
    )
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument(
        "--mri_type",
        default="raw",
        choices=["preprocessed", "raw", "raw_bbox"],
        help="preprocessed=Radiomics/arterial (intensity-normed); raw=Resections_with_rna (resampled to 1×1×3 mm); raw_bbox=raw cropped to tumour bounding box.",
    )
    p.add_argument(
        "--bbox_pad",
        type=int,
        default=10,
        metavar="N",
        help="Voxel padding around the tumour bounding box (raw_bbox mode only). Default: 10.",
    )
    p.add_argument(
        "--split-unit", default="patient", choices=["patient", "slice"],
        help="Unit for train/val split. 'patient' (default) keeps all slices from a patient in one split; 'slice' splits across individual slices.",
    )
    p.add_argument("--val_split", type=float, default=0.1)

    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


if __name__ == "__main__":
    train(_parse_args())
