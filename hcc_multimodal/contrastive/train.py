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

import hcc_multimodal
from hcc_multimodal.contrastive.data import build_dataset
from hcc_multimodal.contrastive.encoders import BACKBONES, GeneEncoder, ImageEncoder
from hcc_multimodal.contrastive.loss import contrastive_loss

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


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    run_dir = _setup_run(args)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    _, weights, _ = BACKBONES[args.model]
    augment = v2.Compose(
        [
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            weights.transforms(),
        ]
    )

    dataset = build_dataset(
        n_per_axis=args.n_per_axis,
        axes=args.axes or None,
        outcome_col=args.outcome_col,
        img_size=args.img_size,
        transform=augment,
        mri_type=args.mri_type,
    )

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

    optimizer = torch.optim.AdamW(
        list(img_enc.parameters()) + list(gene_enc.parameters()),
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
        for imgs, genes, outcomes, _ in train_loader:
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
            step += 1
        scheduler.step()

        img_enc.eval()
        gene_enc.eval()
        total_val = 0.0
        with torch.no_grad():
            for imgs, genes, outcomes, _ in val_loader:
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
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--gene_hidden_dim", type=int, default=256)
    p.add_argument("--freeze_backbone", action="store_true")

    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--lam", type=float, default=0.1)
    p.add_argument(
        "--reg_mode", default="per_modality", choices=["per_modality", "average"]
    )

    p.add_argument("--n_per_axis", type=int, default=10)
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
        default="preprocessed",
        choices=["preprocessed", "raw"],
        help="preprocessed=Radiomics/arterial (intensity-normed); raw=Resections_with_rna (resampled to 1×1×3 mm).",
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
