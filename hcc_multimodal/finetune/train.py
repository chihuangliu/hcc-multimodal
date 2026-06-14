"""DINOv2-style self-supervised finetuning of image backbones on MRI."""

import argparse
import copy
import csv
import json
import subprocess
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import v2
from tqdm import tqdm

import hcc_multimodal
from hcc_multimodal.contrastive.encoders import BACKBONES, BACKBONE_TRANSFORMS, ImageEncoder
from hcc_multimodal.finetune.data import (
    COHORT_CHOICES,
    PHASE_CHOICES,
    MultiCropDataset,
    collect_cohort_patients,
    multicrop_collate,
)
from hcc_multimodal.finetune.loss import DINOLoss

_OUT_ROOT = Path(hcc_multimodal.__file__).parent.parent / "training" / "finetune"
_CONTRASTIVE_ROOT = Path(hcc_multimodal.__file__).parent.parent / "training" / "contrastive"

N_GLOBAL_CROPS = 2


# ---------------------------------------------------------------------------
# DINO projection head
# ---------------------------------------------------------------------------

class DINOHead(nn.Module):
    """MLP projection head with L2-normalised bottleneck and unit-norm last layer.

    Architecture: Linear → BN → GELU → Linear → BN → GELU → Linear(bottleneck)
                  → L2-norm → Linear(out_dim, no bias, weights unit-normalised in forward)

    The last layer's weights are L2-normalised on every forward pass, which is
    equivalent to weight_norm with a frozen unit magnitude (g=1). This avoids
    the deepcopy incompatibility of torch.nn.utils.weight_norm.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 4096,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        w = F.normalize(self.last_layer.weight, dim=1, p=2)
        return F.linear(x, w)


# ---------------------------------------------------------------------------
# Schedule helpers
# ---------------------------------------------------------------------------

def _cosine_schedule(
    base_value: float,
    final_value: float,
    n_steps: int,
    warmup_steps: int = 0,
    warmup_start: float = 0.0,
) -> np.ndarray:
    """Cosine schedule from base_value to final_value over n_steps.

    Optional linear warmup from warmup_start to base_value over warmup_steps.
    """
    warmup = np.linspace(warmup_start, base_value, warmup_steps) if warmup_steps > 0 else np.array([])
    steps = np.arange(n_steps - warmup_steps)
    cosine = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * steps / max(len(steps), 1)))
    return np.concatenate([warmup, cosine])


# ---------------------------------------------------------------------------
# Run setup
# ---------------------------------------------------------------------------

def _setup_run(args: argparse.Namespace, patient_counts: dict[str, int]) -> Path:
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
    meta = {**vars(args), "run_id": run_id, "git_commit": git_hash, "patient_counts": patient_counts}
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    return run_dir


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # --- Collect patients ---
    all_entries: list = []
    cohort_labels: list[int] = []  # for stratified split
    patient_counts: dict[str, int] = {}
    for ci, cohort in enumerate(args.cohorts):
        patients = collect_cohort_patients(cohort, phases=args.phases)
        patient_counts[cohort] = len({p for p, _ in patients})
        print(f"  {cohort}: {patient_counts[cohort]} patients, {len(patients)} volumes ({', '.join(args.phases)})")
        all_entries.extend(patients)
        cohort_labels.extend([ci] * len(patients))
    total = sum(patient_counts.values())
    print(f"  Total: {total} patients across {len(args.cohorts)} cohort(s)")

    if total == 0:
        raise RuntimeError("No patients found. Check data paths.")

    run_dir = _setup_run(args, patient_counts)

    # --- Augmentations ---
    backbone_transform = BACKBONE_TRANSFORMS[args.model]

    global_transform = v2.Compose([
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.ColorJitter(brightness=0.4, contrast=0.4),
        v2.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
        backbone_transform(),
    ])
    local_transform = v2.Compose([
        v2.RandomCrop(args.local_crop_size),
        v2.Resize((args.img_size, args.img_size), antialias=True),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.ColorJitter(brightness=0.4, contrast=0.4),
        v2.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0)),
        backbone_transform(),
    ])

    # --- Dataset & split (patient-level, stratified by cohort) ---
    dataset = MultiCropDataset(
        entries=all_entries,
        n_per_axis=args.n_per_axis,
        axes=args.axes or [0],
        img_size=args.img_size,
        n_local_crops=args.n_local_crops,
        global_transform=global_transform,
        local_transform=local_transform,
    )

    pid_to_cohort: dict[int, int] = {
        pid: ci
        for ci, cohort in enumerate(args.cohorts)
        for pid, _ in collect_cohort_patients(cohort, phases=args.phases)
    }
    all_pids = sorted({pid for pid, _, _, _ in dataset._index})
    labels_for_split = [pid_to_cohort.get(p, 0) for p in all_pids]

    if len(set(labels_for_split)) > 1 and len(all_pids) >= 4:
        train_pids, val_pids = train_test_split(
            all_pids,
            test_size=args.val_split,
            stratify=labels_for_split,
            random_state=args.seed,
        )
    else:
        split_idx = max(1, int(len(all_pids) * (1 - args.val_split)))
        train_pids, val_pids = all_pids[:split_idx], all_pids[split_idx:]

    train_pid_set, val_pid_set = set(train_pids), set(val_pids)
    train_idx = [i for i, (pid, _, _, _) in enumerate(dataset._index) if pid in train_pid_set]
    val_idx = [i for i, (pid, _, _, _) in enumerate(dataset._index) if pid in val_pid_set]

    pin = device.type == "cuda"
    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin,
        drop_last=True,
        collate_fn=multicrop_collate,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
        collate_fn=multicrop_collate,
    )

    # --- Models ---
    student_enc = ImageEncoder(args.model, args.embed_dim, freeze=False).to(device)
    student_head = DINOHead(
        in_dim=args.embed_dim,
        out_dim=args.out_dim,
        hidden_dim=args.dino_hidden_dim,
        bottleneck_dim=args.dino_bottleneck_dim,
    ).to(device)

    teacher_enc = copy.deepcopy(student_enc).to(device)
    teacher_head = copy.deepcopy(student_head).to(device)
    for p in teacher_enc.parameters():
        p.requires_grad_(False)
    for p in teacher_head.parameters():
        p.requires_grad_(False)

    # Optional warm-start from a previous finetune or contrastive checkpoint
    if args.base_model is not None:
        for search_root in [_OUT_ROOT, _CONTRASTIVE_ROOT]:
            ckpt_path = search_root / args.base_model / "best_model.pt"
            if ckpt_path.exists():
                break
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        student_enc.load_state_dict(ckpt["img_enc"])
        teacher_enc.load_state_dict(ckpt["img_enc"])
        print(f"Loaded img_enc weights from {ckpt_path}")

    # --- Optimizer & schedules ---
    params = list(student_enc.parameters()) + list(student_head.parameters())
    trainable = [p for p in params if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    n_steps = args.epochs * len(train_loader)
    warmup_steps = args.warmup_epochs * len(train_loader)
    lr_schedule = _cosine_schedule(
        args.lr, args.min_lr, n_steps, warmup_steps=warmup_steps, warmup_start=0.0
    )
    wd_schedule = _cosine_schedule(
        args.weight_decay, args.weight_decay_end, n_steps
    )
    momentum_schedule = _cosine_schedule(
        args.momentum_base, 1.0, n_steps
    )

    dino_loss = DINOLoss(
        out_dim=args.out_dim,
        student_temp=args.student_temp,
        center_momentum=0.9,
    ).to(device)

    # --- Training loop ---
    csv_path = run_dir / "losses.csv"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss"])

    best_val = float("inf")
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        # Teacher temperature schedule (linear warmup)
        if epoch <= args.warmup_teacher_temp_epochs:
            teacher_temp = (
                args.teacher_temp_start
                + (args.teacher_temp - args.teacher_temp_start)
                * (epoch - 1) / max(args.warmup_teacher_temp_epochs - 1, 1)
            )
        else:
            teacher_temp = args.teacher_temp

        student_enc.train()
        student_head.train()
        total_train = 0.0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{args.epochs} train", leave=False)
        for views, _ in train_pbar:
            # Update LR and WD
            lr = float(lr_schedule[min(global_step, len(lr_schedule) - 1)])
            wd = float(wd_schedule[min(global_step, len(wd_schedule) - 1)])
            for g in optimizer.param_groups:
                g["lr"] = lr
                g["weight_decay"] = wd

            views = [v.to(device) for v in views]

            # Teacher forward (global crops only, no grad)
            with torch.no_grad():
                teacher_out = [
                    teacher_head(teacher_enc(views[i]))
                    for i in range(N_GLOBAL_CROPS)
                ]

            # Student forward (all crops)
            student_out = [student_head(student_enc(v)) for v in views]

            loss = dino_loss(student_out, teacher_out, teacher_temp)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=3.0)
            optimizer.step()

            # EMA teacher update
            m = float(momentum_schedule[min(global_step, len(momentum_schedule) - 1)])
            with torch.no_grad():
                for p_s, p_t in zip(
                    list(student_enc.parameters()) + list(student_head.parameters()),
                    list(teacher_enc.parameters()) + list(teacher_head.parameters()),
                ):
                    p_t.data.mul_(m).add_((1 - m) * p_s.detach().data)

            dino_loss.update_center(teacher_out)

            total_train += loss.item()
            train_pbar.set_postfix(loss=f"{loss.item():.4f}", teacher_temp=f"{teacher_temp:.4f}")
            global_step += 1

        # Validation
        student_enc.eval()
        student_head.eval()
        total_val = 0.0
        with torch.no_grad():
            for views, _ in tqdm(val_loader, desc=f"Epoch {epoch:3d}/{args.epochs} val  ", leave=False):
                views = [v.to(device) for v in views]
                teacher_out = [teacher_head(teacher_enc(views[i])) for i in range(N_GLOBAL_CROPS)]
                student_out = [student_head(student_enc(v)) for v in views]
                total_val += dino_loss(student_out, teacher_out, teacher_temp).item()

        avg_train = total_train / len(train_loader)
        avg_val = total_val / max(1, len(val_loader))
        print(f"Epoch {epoch:3d}/{args.epochs}  train={avg_train:.4f}  val={avg_val:.4f}  teacher_temp={teacher_temp:.4f}")

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, avg_train, avg_val])

        checkpoint = {
            "img_enc": student_enc.state_dict(),
            "teacher_img_enc": teacher_enc.state_dict(),
            "dino_head": student_head.state_dict(),
        }
        if avg_val < best_val:
            best_val = avg_val
            torch.save(checkpoint, run_dir / "best_model.pt")

    if args.epochs > 0:
        torch.save(checkpoint, run_dir / "last_model.pt")
        print(f"Done. Best val loss: {best_val:.4f}  →  {run_dir}")
    else:
        print(f"Run dir: {run_dir} (no training, --epochs 0)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DINOv2-style self-supervised MRI finetuning")

    # Cohorts
    p.add_argument(
        "--cohorts",
        nargs="+",
        required=True,
        choices=COHORT_CHOICES,
        help="Which patient cohorts to use. Can be one or more of: resection, soramic, lausanne.",
    )
    p.add_argument(
        "--phases",
        nargs="+",
        default=["arterial"],
        choices=PHASE_CHOICES,
        help=(
            "MRI phases to include per patient. Each available phase becomes a separate volume. "
            "Default: arterial only. Options: arterial, portovenous, delayed, hpb. "
            "Phases not present for a cohort or missing on disk are silently skipped."
        ),
    )

    # Backbone
    p.add_argument("--model", default="dinov2_vitb14", choices=list(BACKBONES))
    p.add_argument(
        "--base_model",
        default=None,
        metavar="RUN_ID",
        help="Run ID to warm-start img_enc from (searches training/finetune/ then training/contrastive/).",
    )
    p.add_argument("--embed_dim", type=int, default=128)

    # DINO head
    p.add_argument("--out_dim", type=int, default=4096, help="Prototype dimension K.")
    p.add_argument("--dino_hidden_dim", type=int, default=2048)
    p.add_argument("--dino_bottleneck_dim", type=int, default=256)

    # DINO loss
    p.add_argument("--student_temp", type=float, default=0.1)
    p.add_argument("--teacher_temp", type=float, default=0.07)
    p.add_argument("--teacher_temp_start", type=float, default=0.04, help="Initial teacher temp before warmup.")
    p.add_argument("--warmup_teacher_temp_epochs", type=int, default=5)

    # Multi-crop
    p.add_argument("--n_local_crops", type=int, default=6, help="Number of local crops per slice.")
    p.add_argument("--local_crop_size", type=int, default=96, help="Pixel size of local random crops.")

    # EMA momentum
    p.add_argument("--momentum_base", type=float, default=0.9, help="Starting EMA momentum (cosine → 1.0).")

    # Data
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
        default=[0],
        metavar="AXIS",
        help="Axes to slice (0=sagittal 1=coronal 2=axial). Default: 0.",
    )
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--val_split", type=float, default=0.1)

    # Optimiser
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--warmup_epochs", type=int, default=5, help="LR warmup epochs.")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min_lr", type=float, default=1e-6, help="Final LR after cosine decay.")
    p.add_argument("--weight_decay", type=float, default=0.04)
    p.add_argument("--weight_decay_end", type=float, default=0.4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


if __name__ == "__main__":
    train(_parse_args())
