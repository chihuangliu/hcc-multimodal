"""Refit a GeneEncoder against a frozen, pre-trained image encoder.

Motivation: a trained contrastive model's GeneEncoder expects genes in the
column order produced at training time. That order came from non-deterministic
Python `set` iteration (`_load_gene_matrix`) and was never persisted, so the
gene -> input-slot mapping of an existing `gene_enc` is unrecoverable. With the
gene order now pinned (sorted) in `_load_gene_matrix`, this script rebuilds a
GeneEncoder with a KNOWN order while keeping the source model's IMAGE encoder
frozen and unchanged.

Because the downstream RFS head consumes image embeddings only, freezing the
image encoder means downstream performance is identical to the source model by
construction. The refit gene branch is re-aligned to the same fixed image space
(not co-trained), giving an interpretable, deterministic gene encoder for IG.

The source model's config (gene set, slices, mri_type, loss, optimizer, split)
is read from its metadata.json so the refit matches the original training setup.
A new run id (uuid) is created; the source run is never overwritten. The new
run's best_model.pt reuses the source image encoder verbatim and stores the
refit gene encoder; cached image embeddings are copied so downstream eval and
the IG script see the exact same z_img.

Example:
    python scripts/refit_gene_encoder.py --src-model 9109a6c2
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import v2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hcc_multimodal.contrastive.data import build_dataset
from hcc_multimodal.contrastive.encoders import BACKBONE_TRANSFORMS, GeneEncoder, ImageEncoder
from hcc_multimodal.contrastive.loss import contrastive_loss
from hcc_multimodal.contrastive.train import _GENE_SETS

OUT_ROOT = Path("training/contrastive")


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run(args) -> None:
    src_dir = OUT_ROOT / args.src_model
    meta = json.loads((src_dir / "metadata.json").read_text())
    device = _pick_device()
    torch.manual_seed(args.seed)

    # --- dataset, identical setup to the source training run -----------------
    augment = v2.Compose([
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        BACKBONE_TRANSFORMS[meta["model"]](),
    ])
    genes = set(_GENE_SETS[meta["gene_set"]])
    axes = meta["axes"] if isinstance(meta["axes"], list) else [meta["axes"]]
    dataset = build_dataset(
        n_per_axis=meta["n_per_axis"],
        axes=axes,
        outcome_col=meta["outcome_col"],
        img_size=meta["img_size"],
        transform=augment,
        mri_type=meta["mri_type"],
        genes=genes,
        bbox_pad=meta["bbox_pad"],
    )
    gene_order = list(dataset.gene_matrix.columns)
    print(f"Gene order ({len(gene_order)}): {gene_order}")

    # --- train/val split, matching train.py ----------------------------------
    if meta["split_unit"] == "patient":
        patients = sorted({pid for pid, _, _ in dataset._index})
        labels = [int(dataset.outcomes[pid]) for pid in patients]
        tr_pids, va_pids = train_test_split(
            patients, test_size=meta["val_split"], stratify=labels, random_state=args.seed
        )
        tr_set, va_set = set(tr_pids), set(va_pids)
        train_idx = [i for i, (pid, _, _) in enumerate(dataset._index) if pid in tr_set]
        val_idx = [i for i, (pid, _, _) in enumerate(dataset._index) if pid in va_set]
    else:
        labels = [int(dataset.outcomes[pid]) for pid, _, _ in dataset._index]
        train_idx, val_idx = train_test_split(
            list(range(len(dataset))), test_size=meta["val_split"],
            stratify=labels, random_state=args.seed
        )
    train_ds, val_ds = Subset(dataset, train_idx), Subset(dataset, val_idx)
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=meta["batch_size"], shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=meta["batch_size"], shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin)

    # --- frozen source image encoder + fresh gene encoder --------------------
    img_enc = ImageEncoder(meta["model"], meta["embed_dim"], meta["freeze_backbone"]).to(device)
    src_ckpt = torch.load(src_dir / "best_model.pt", map_location=device, weights_only=False)
    img_enc.load_state_dict(src_ckpt["img_enc"])
    img_enc.eval()
    for p in img_enc.parameters():
        p.requires_grad_(False)

    gene_dim = dataset.gene_matrix.shape[1]
    gene_enc = GeneEncoder(gene_dim, meta["gene_hidden_dim"], meta["embed_dim"]).to(device)

    optimizer = torch.optim.AdamW(gene_enc.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # --- new run dir ----------------------------------------------------------
    run_id = uuid.uuid4().hex[:8]
    run_dir = OUT_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "losses.csv"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss"])

    best_val = float("inf")
    best_state = None
    step = 0
    for epoch in range(1, args.epochs + 1):
        gene_enc.train()
        total_train = 0.0
        for imgs, gvecs, outcomes, _ in train_loader:
            imgs, gvecs, outcomes = imgs.to(device), gvecs.to(device), outcomes.to(device)
            with torch.no_grad():
                z_img = img_enc(imgs)
            loss = contrastive_loss(
                z_img, gene_enc(gvecs), outcomes=outcomes,
                temperature=meta["temperature"], lam=meta["lam"], step=step,
                reg_mode=meta["reg_mode"],
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train += loss.item()
            step += 1
        scheduler.step()

        gene_enc.eval()
        total_val = 0.0
        with torch.no_grad():
            for imgs, gvecs, outcomes, _ in val_loader:
                imgs, gvecs, outcomes = imgs.to(device), gvecs.to(device), outcomes.to(device)
                z_img = img_enc(imgs)
                total_val += contrastive_loss(
                    z_img, gene_enc(gvecs), outcomes=outcomes,
                    temperature=meta["temperature"], lam=meta["lam"], step=step,
                    reg_mode=meta["reg_mode"],
                ).item()

        avg_train = total_train / len(train_loader)
        avg_val = total_val / max(1, len(val_loader))
        print(f"Epoch {epoch:3d}/{args.epochs}  train={avg_train:.4f}  val={avg_val:.4f}")
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, avg_train, avg_val])
        if avg_val < best_val:
            best_val = avg_val
            best_state = {k: v.detach().cpu().clone() for k, v in gene_enc.state_dict().items()}

    # --- save new run: source img_enc verbatim + refit gene_enc --------------
    img_state = {k: v.detach().cpu().clone() for k, v in img_enc.state_dict().items()}
    torch.save({"img_enc": img_state, "gene_enc": best_state}, run_dir / "best_model.pt")
    torch.save({"img_enc": img_state, "gene_enc": gene_enc.state_dict()}, run_dir / "last_model.pt")

    new_meta = {
        **meta,
        "run_id": run_id,
        "git_commit": _git_hash(),
        "refit_gene_enc_from": args.src_model,
        "deterministic_gene_order": True,
        "gene_order": gene_order,
        "refit_seed": args.seed,
    }
    (run_dir / "metadata.json").write_text(json.dumps(new_meta, indent=2))

    # Copy cached image embeddings (img_enc is identical -> z_img identical),
    # so downstream eval and the IG script reuse the source's exact embeddings.
    src_cache = src_dir / "cached_embeddings"
    if src_cache.exists():
        shutil.copytree(src_cache, run_dir / "cached_embeddings", dirs_exist_ok=True)

    print(f"\nDone. Best val loss: {best_val:.4f}")
    print(f"New run id: {run_id}  ->  {run_dir}")
    print(f"Image encoder copied verbatim from {args.src_model} "
          f"(downstream performance identical by construction).")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src-model", default="9109a6c2", help="Source run id (frozen image encoder)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
