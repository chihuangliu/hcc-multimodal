"""DINOv2/v3-style self-supervised finetuning of image backbones on MRI.

Checkpoints save only the backbone (not a contrastive projection head), so
they can be loaded cleanly by contrastive/train.py via --base_model <run_id>.

--base_model accepts either:
  - a model name from the BACKBONES registry (e.g. 'dinov2_vitb14') → pretrained init
  - a run ID from training/finetune/dino/, training/finetune/, or
    training/contrastive/ → warm-start backbone from that checkpoint
"""

import argparse
import copy
import csv
import json
import os
import subprocess
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset
from torchvision.transforms import v2
from tqdm import tqdm

import hcc_multimodal
from hcc_multimodal.contrastive.encoders import BACKBONES, BACKBONE_TRANSFORMS
from hcc_multimodal.finetune.data import (
    COHORT_CHOICES,
    PHASE_CHOICES,
    MultiCropDataset,
    collect_cohort_patients,
    multicrop_collate,
)
from hcc_multimodal.finetune.loss import DINOLoss, DINOHead

_OUT_ROOT = Path(hcc_multimodal.__file__).parent.parent / "training" / "finetune" / "dino"
_LEGACY_FINETUNE_ROOT = Path(hcc_multimodal.__file__).parent.parent / "training" / "finetune"
_CONTRASTIVE_ROOT = Path(hcc_multimodal.__file__).parent.parent / "training" / "contrastive"

N_GLOBAL_CROPS = 2


# ---------------------------------------------------------------------------
# Base model resolution
# ---------------------------------------------------------------------------

def resolve_base_model(base_model: str) -> tuple[str, Path | None]:
    """Return (model_name, checkpoint_path_or_None).

    Accepts a BACKBONES key (pretrained init) or a run ID (checkpoint init).
    Run ID search order: finetune/dino/ → finetune/ → contrastive/.
    """
    if base_model in BACKBONES:
        return base_model, None

    for root in [_OUT_ROOT, _LEGACY_FINETUNE_ROOT, _CONTRASTIVE_ROOT]:
        meta_path = root / base_model / "metadata.json"
        if meta_path.exists():
            model_name = json.loads(meta_path.read_text())["model"]
            return model_name, root / base_model / "best_model.pt"

    raise ValueError(
        f"{base_model!r} is not a BACKBONES key or a known run ID. "
        f"Known models: {sorted(BACKBONES)}"
    )


def _load_backbone_weights(backbone, ckpt_path: Path, device: torch.device) -> None:
    """Load backbone weights from either new (backbone key) or legacy (img_enc key) checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "backbone" in ckpt:
        backbone.load_state_dict(ckpt["backbone"])
    else:
        # legacy finetune or contrastive checkpoint: img_enc contains backbone.* and proj.*
        state = {
            k[len("backbone."):]: v
            for k, v in ckpt["img_enc"].items()
            if k.startswith("backbone.")
        }
        backbone.load_state_dict(state)
    print(f"Loaded backbone weights from {ckpt_path}")


# ---------------------------------------------------------------------------
# Backbone forward helpers
# ---------------------------------------------------------------------------

def _backbone_forward(backbone, x: torch.Tensor, return_patches: bool = False):
    """Return (cls_token, patch_tokens_or_None).

    patch_tokens are L2-normalised and only returned when return_patches=True.
    Gram anchoring requires a HuggingFace ViT backbone (dinov2_* / dinov3_*).
    """
    # Unwrap DDP if needed so we can introspect the underlying module.
    raw = backbone.module if isinstance(backbone, DDP) else backbone
    if hasattr(raw, "model"):  # _HFViTWrapper
        out = raw.model(x)
        cls = out.last_hidden_state[:, 0]
        patches = F.normalize(out.last_hidden_state[:, 1:], dim=-1) if return_patches else None
        return cls, patches
    # torchvision ViT — heads already replaced with Identity
    if return_patches:
        raise ValueError("--gram-anchoring requires a HuggingFace ViT backbone (dinov2_* or dinov3_*)")
    return backbone(x), None


def _gram_loss(student_patches: torch.Tensor, gram_patches: torch.Tensor) -> torch.Tensor:
    """||G_S - G_G||_F^2 where G = X @ X^T on L2-normalised patch features."""
    G_s = torch.bmm(student_patches, student_patches.transpose(1, 2))
    G_g = torch.bmm(gram_patches, gram_patches.transpose(1, 2))
    return ((G_s - G_g) ** 2).mean()


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
    warmup = np.linspace(warmup_start, base_value, warmup_steps) if warmup_steps > 0 else np.array([])
    steps = np.arange(n_steps - warmup_steps)
    cosine = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * steps / max(len(steps), 1)))
    return np.concatenate([warmup, cosine])


# ---------------------------------------------------------------------------
# Run setup
# ---------------------------------------------------------------------------

def _setup_run(args: argparse.Namespace, model_name: str, patient_counts: dict[str, int]) -> Path:
    run_id = uuid.uuid4().hex[:8]
    run_dir = _OUT_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        git_hash = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
            .decode().strip()
        )
    except Exception:
        git_hash = "unknown"
    meta = {**vars(args), "model": model_name, "run_id": run_id, "git_commit": git_hash, "patient_counts": patient_counts}
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    return run_dir


# ---------------------------------------------------------------------------
# Per-rank training worker
# ---------------------------------------------------------------------------

def _train_worker(
    rank: int,
    world_size: int,
    args: argparse.Namespace,
    run_dir: Path,
    model_name: str,
    ckpt_path: Path | None,
) -> None:
    use_ddp = world_size > 1
    is_main = rank == 0

    if use_ddp:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    torch.manual_seed(args.seed)

    # --- Collect patients (fast; each rank does this independently) ---
    all_entries: list = []
    cohort_labels: list[int] = []
    for ci, cohort in enumerate(args.cohorts):
        patients = collect_cohort_patients(cohort, phases=args.phases)
        all_entries.extend(patients)
        cohort_labels.extend([ci] * len(patients))

    # --- Augmentations ---
    backbone_transform = BACKBONE_TRANSFORMS[model_name]
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

    # --- Dataset & split ---
    dataset = MultiCropDataset(
        entries=all_entries,
        n_per_axis=args.n_per_axis,
        axes=args.axes or [0],
        img_size=args.img_size,
        n_local_crops=args.n_local_crops,
        global_transform=global_transform,
        local_transform=local_transform,
        cache_size=args.cache_size,
    )
    pid_to_cohort = {
        pid: ci
        for ci, cohort in enumerate(args.cohorts)
        for pid, _ in collect_cohort_patients(cohort, phases=args.phases)
    }
    all_pids = sorted({pid for pid, _, _, _ in dataset._index})
    labels_for_split = [pid_to_cohort.get(p, 0) for p in all_pids]

    if len(set(labels_for_split)) > 1 and len(all_pids) >= 4:
        train_pids, val_pids = train_test_split(
            all_pids, test_size=args.val_split, stratify=labels_for_split, random_state=args.seed
        )
    else:
        split_idx = max(1, int(len(all_pids) * (1 - args.val_split)))
        train_pids, val_pids = all_pids[:split_idx], all_pids[split_idx:]

    train_pid_set, val_pid_set = set(train_pids), set(val_pids)
    train_idx = [i for i, (pid, _, _, _) in enumerate(dataset._index) if pid in train_pid_set]
    val_idx = [i for i, (pid, _, _, _) in enumerate(dataset._index) if pid in val_pid_set]

    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    pin = device.type == "cuda"
    if use_ddp:
        train_sampler = DistributedSampler(train_subset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed)
        val_sampler = DistributedSampler(val_subset, num_replicas=world_size, rank=rank, shuffle=False)
        train_loader = DataLoader(
            train_subset, batch_size=args.batch_size, sampler=train_sampler,
            num_workers=args.num_workers, pin_memory=pin, drop_last=True, collate_fn=multicrop_collate,
        )
        val_loader = DataLoader(
            val_subset, batch_size=args.batch_size, sampler=val_sampler,
            num_workers=args.num_workers, pin_memory=pin, collate_fn=multicrop_collate,
        )
    else:
        train_sampler = None
        train_loader = DataLoader(
            train_subset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=pin, drop_last=True, collate_fn=multicrop_collate,
        )
        val_loader = DataLoader(
            val_subset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=pin, collate_fn=multicrop_collate,
        )

    # --- Models ---
    student_backbone, feat_dim = BACKBONES[model_name]()
    student_backbone = student_backbone.to(device)
    student_head = DINOHead(
        in_dim=feat_dim,
        out_dim=args.out_dim,
        hidden_dim=args.dino_hidden_dim,
        bottleneck_dim=args.dino_bottleneck_dim,
    ).to(device)

    teacher_backbone = copy.deepcopy(student_backbone).to(device)
    teacher_head = copy.deepcopy(student_head).to(device)
    for p in teacher_backbone.parameters():
        p.requires_grad_(False)
    for p in teacher_head.parameters():
        p.requires_grad_(False)

    if ckpt_path is not None:
        _load_backbone_weights(student_backbone, ckpt_path, device)
        _load_backbone_weights(teacher_backbone, ckpt_path, device)

    if args.gram_anchoring:
        gram_backbone = copy.deepcopy(teacher_backbone).to(device)
        for p in gram_backbone.parameters():
            p.requires_grad_(False)
    else:
        gram_backbone = None

    if use_ddp:
        # Sync BN stats across ranks for the student head's BatchNorm layers.
        student_head = nn.SyncBatchNorm.convert_sync_batchnorm(student_head)
        student_backbone = DDP(student_backbone, device_ids=[rank])
        student_head = DDP(student_head, device_ids=[rank])

    # Raw (unwrapped) references for EMA updates and checkpoint saving.
    raw_student_backbone = student_backbone.module if use_ddp else student_backbone
    raw_student_head = student_head.module if use_ddp else student_head

    # --- Optimizer & schedules ---
    trainable = [p for p in list(raw_student_backbone.parameters()) + list(raw_student_head.parameters()) if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    n_steps = args.epochs * len(train_loader)
    warmup_steps = args.warmup_epochs * len(train_loader)
    lr_schedule = _cosine_schedule(args.lr, args.min_lr, n_steps, warmup_steps=warmup_steps)
    wd_schedule = _cosine_schedule(args.weight_decay, args.weight_decay_end, n_steps)
    momentum_schedule = _cosine_schedule(args.momentum_base, 1.0, n_steps)

    dino_loss = DINOLoss(out_dim=args.out_dim, student_temp=args.student_temp, center_momentum=0.9).to(device)

    # --- Training loop ---
    csv_path = run_dir / "losses.csv"
    if is_main:
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_loss"])

    best_val = float("inf")
    global_step = 0
    checkpoint = {}

    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if epoch <= args.warmup_teacher_temp_epochs:
            teacher_temp = (
                args.teacher_temp_start
                + (args.teacher_temp - args.teacher_temp_start)
                * (epoch - 1) / max(args.warmup_teacher_temp_epochs - 1, 1)
            )
        else:
            teacher_temp = args.teacher_temp

        student_backbone.train()
        student_head.train()
        total_train = 0.0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{args.epochs} train", leave=False, disable=not is_main)
        for views, _ in train_pbar:
            lr = float(lr_schedule[min(global_step, len(lr_schedule) - 1)])
            wd = float(wd_schedule[min(global_step, len(wd_schedule) - 1)])
            for g in optimizer.param_groups:
                g["lr"] = lr
                g["weight_decay"] = wd

            views = [v.to(device) for v in views]

            with torch.no_grad():
                teacher_out = [
                    teacher_head(_backbone_forward(teacher_backbone, views[i])[0])
                    for i in range(N_GLOBAL_CROPS)
                ]

            student_cls = [_backbone_forward(student_backbone, v, return_patches=args.gram_anchoring) for v in views]
            student_out = [student_head(cls) for cls, _ in student_cls]

            loss = dino_loss(student_out, teacher_out, teacher_temp)

            if args.gram_anchoring:
                with torch.no_grad():
                    gram_patches_list = [
                        _backbone_forward(gram_backbone, views[i], return_patches=True)[1]
                        for i in range(N_GLOBAL_CROPS)
                    ]
                for i in range(N_GLOBAL_CROPS):
                    _, s_patches = student_cls[i]
                    loss = loss + args.gram_weight * _gram_loss(s_patches, gram_patches_list[i])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=3.0)
            optimizer.step()

            # EMA teacher update. DDP keeps all students identical via gradient all-reduce,
            # so applying EMA locally on each rank produces identical teachers.
            m = float(momentum_schedule[min(global_step, len(momentum_schedule) - 1)])
            with torch.no_grad():
                for p_s, p_t in zip(
                    list(raw_student_backbone.parameters()) + list(raw_student_head.parameters()),
                    list(teacher_backbone.parameters()) + list(teacher_head.parameters()),
                ):
                    p_t.data.mul_(m).add_((1 - m) * p_s.detach().data)

            if args.gram_anchoring and (global_step + 1) % args.gram_update_interval == 0:
                with torch.no_grad():
                    for p_g, p_t in zip(gram_backbone.parameters(), teacher_backbone.parameters()):
                        p_g.data.copy_(p_t.data)

            # Center update: all-reduce the batch mean across ranks so every rank's
            # centering buffer stays identical despite seeing different data shards.
            if use_ddp:
                batch_center = torch.cat(teacher_out, dim=0).mean(dim=0)
                dist.all_reduce(batch_center, op=dist.ReduceOp.AVG)
                dino_loss.center.mul_(dino_loss.center_momentum).add_(
                    batch_center.unsqueeze(0) * (1 - dino_loss.center_momentum)
                )
            else:
                dino_loss.update_center(teacher_out)

            total_train += loss.item()
            train_pbar.set_postfix(loss=f"{loss.item():.4f}", teacher_temp=f"{teacher_temp:.4f}")
            global_step += 1

        # Validation
        student_backbone.eval()
        student_head.eval()
        total_val = 0.0
        with torch.no_grad():
            for views, _ in tqdm(val_loader, desc=f"Epoch {epoch:3d}/{args.epochs} val  ", leave=False, disable=not is_main):
                views = [v.to(device) for v in views]
                teacher_out = [teacher_head(_backbone_forward(teacher_backbone, views[i])[0]) for i in range(N_GLOBAL_CROPS)]
                student_out = [student_head(_backbone_forward(student_backbone, v)[0]) for v in views]
                total_val += dino_loss(student_out, teacher_out, teacher_temp).item()

        # All-reduce per-step averages so rank 0 logs the global mean.
        if use_ddp:
            metrics = torch.tensor(
                [total_train / len(train_loader), total_val / max(1, len(val_loader))],
                device=device,
            )
            dist.all_reduce(metrics, op=dist.ReduceOp.AVG)
            avg_train, avg_val = metrics[0].item(), metrics[1].item()
        else:
            avg_train = total_train / len(train_loader)
            avg_val = total_val / max(1, len(val_loader))

        if is_main:
            print(f"Epoch {epoch:3d}/{args.epochs}  train={avg_train:.4f}  val={avg_val:.4f}  teacher_temp={teacher_temp:.4f}")
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow([epoch, avg_train, avg_val])

            checkpoint = {
                "backbone": raw_student_backbone.state_dict(),
                "teacher_backbone": teacher_backbone.state_dict(),
                "dino_head": raw_student_head.state_dict(),
                "model": model_name,
            }
            if avg_val < best_val:
                best_val = avg_val
                torch.save(checkpoint, run_dir / "best_model.pt")

    if is_main and args.epochs > 0:
        torch.save(checkpoint, run_dir / "last_model.pt")
        print(f"Done. Best val loss: {best_val:.4f}  →  {run_dir}")
    elif is_main:
        print(f"Run dir: {run_dir} (no training, --epochs 0)")

    if use_ddp:
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    if args.ngpu > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("--ngpu > 1 requires CUDA")
        if args.ngpu > torch.cuda.device_count():
            raise RuntimeError(
                f"--ngpu {args.ngpu} requested but only {torch.cuda.device_count()} CUDA device(s) available"
            )

    torch.manual_seed(args.seed)

    model_name, ckpt_path = resolve_base_model(args.base_model)

    all_entries: list = []
    patient_counts: dict[str, int] = {}
    cohort_labels: list[int] = []
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

    run_dir = _setup_run(args, model_name, patient_counts)

    worker_args = (args.ngpu, args, run_dir, model_name, ckpt_path)
    if args.ngpu > 1:
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "12355")
        mp.spawn(_train_worker, args=worker_args, nprocs=args.ngpu, join=True)
    else:
        _train_worker(0, *worker_args)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DINOv2/v3-style self-supervised MRI backbone finetuning")

    p.add_argument("--cohorts", nargs="+", required=True, choices=COHORT_CHOICES)
    p.add_argument("--phases", nargs="+", default=["arterial"], choices=PHASE_CHOICES)

    p.add_argument(
        "--base_model",
        required=True,
        metavar="MODEL_OR_RUN_ID",
        help=(
            "Starting point. Either a model name from the BACKBONES registry "
            "(e.g. 'dinov2_vitb14') for pretrained init, or a run ID "
            "(e.g. '84f180d9') to warm-start from a previous checkpoint."
        ),
    )

    p.add_argument("--out_dim", type=int, default=4096)
    p.add_argument("--dino_hidden_dim", type=int, default=2048)
    p.add_argument("--dino_bottleneck_dim", type=int, default=256)

    p.add_argument("--student_temp", type=float, default=0.1)
    p.add_argument("--teacher_temp", type=float, default=0.07)
    p.add_argument("--teacher_temp_start", type=float, default=0.04)
    p.add_argument("--warmup_teacher_temp_epochs", type=int, default=5)

    p.add_argument("--n_local_crops", type=int, default=6)
    p.add_argument("--local_crop_size", type=int, default=96)
    p.add_argument("--momentum_base", type=float, default=0.9)

    # DINOv3 Gram anchoring
    p.add_argument("--gram-anchoring", action="store_true", help="Enable DINOv3-style Gram anchoring loss.")
    p.add_argument("--gram-weight", type=float, default=1.0, dest="gram_weight")
    p.add_argument("--gram-update-interval", type=int, default=10000, dest="gram_update_interval",
                   help="Steps between Gram teacher syncs to EMA teacher.")

    p.add_argument("--n_per_axis", type=lambda v: None if v == "all" else int(v), default=None, metavar="N|all")
    p.add_argument("--axes", type=int, nargs="+", default=[0], metavar="AXIS")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--val_split", type=float, default=0.1)

    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=0.04)
    p.add_argument("--weight_decay_end", type=float, default=0.4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--cache_size", type=int, default=64, metavar="N",
                   help="Resampled volumes kept in each worker's LRU cache (0 disables). "
                        "Approx RAM = ~50MB x N x num_workers.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ngpu", type=int, default=1, metavar="N",
                   help="Number of GPUs for data-parallel training (default: 1).")

    return p.parse_args()


if __name__ == "__main__":
    train(_parse_args())
