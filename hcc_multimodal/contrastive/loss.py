from typing import Callable, Literal

import torch
import torch.nn.functional as F


def nt_xent_loss(
    z_img: torch.Tensor,
    z_gene: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """NT-Xent loss aligning image and gene embeddings per patient.

    Args:
        z_img: (N, d) image embeddings
        z_gene: (N, d) gene embeddings
        temperature: softmax temperature

    Returns:
        Scalar loss
    """
    z_img = F.normalize(z_img, dim=-1)
    z_gene = F.normalize(z_gene, dim=-1)

    N = z_img.shape[0]
    z = torch.cat([z_img, z_gene], dim=0)          # (2N, d)
    sim = torch.mm(z, z.T) / temperature            # (2N, 2N)

    mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float("-inf"))

    labels = torch.arange(N, device=z.device)
    labels = torch.cat([labels + N, labels])

    return F.cross_entropy(sim, labels)


def _outcome_reg(
    z: torch.Tensor,
    outcomes: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    z = F.normalize(z, dim=-1)
    sim = torch.mm(z, z.T) / temperature
    o = outcomes.unsqueeze(0)
    same = (o == o.T).float()
    same.masked_fill_(torch.eye(z.shape[0], dtype=torch.bool, device=z.device), 0)
    return ((1 - sim) * same).sum() / same.sum().clamp(min=1)


def _reg_per_modality(
    z_img: torch.Tensor,
    z_gene: torch.Tensor,
    outcomes: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Regularization applied independently to each modality."""
    return _outcome_reg(z_img, outcomes, temperature) + _outcome_reg(z_gene, outcomes, temperature)


def _reg_average(
    z_img: torch.Tensor,
    z_gene: torch.Tensor,
    outcomes: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Regularization on the averaged cross-modal patient representation.

    Assumes the two modalities are already well-aligned; prefer
    'per_modality' when alignment quality is uncertain.
    """
    z = (F.normalize(z_img, dim=-1) + F.normalize(z_gene, dim=-1)) / 2
    return _outcome_reg(z, outcomes, temperature)


def contrastive_loss(
    z_img: torch.Tensor,
    z_gene: torch.Tensor,
    outcomes: torch.Tensor | None = None,
    temperature: float = 0.07,
    lam: float | Callable[[int], float] = 0.1,
    step: int = 0,
    reg_mode: Literal["per_modality", "average"] = "per_modality",
) -> torch.Tensor:
    """Combined NT-Xent + outcome regularization loss.

    Args:
        z_img: (N, d) image embeddings
        z_gene: (N, d) gene embeddings
        outcomes: (N,) integer outcome labels; if None, skips regularization
        temperature: softmax temperature
        lam: regularization weight — either a fixed float or a callable
            lam(step) -> float for warmup/scheduling
        step: current training step, passed to lam if it is a callable
        reg_mode: 'per_modality' applies regularization independently to each
            modality (recommended when alignment is uncertain); 'average' uses
            the averaged cross-modal embedding (assumes alignment)

    Returns:
        Scalar total loss
    """
    loss = nt_xent_loss(z_img, z_gene, temperature)

    if outcomes is not None:
        weight = lam(step) if callable(lam) else lam
        if reg_mode == "per_modality":
            reg = _reg_per_modality(z_img, z_gene, outcomes, temperature)
        else:
            reg = _reg_average(z_img, z_gene, outcomes, temperature)
        loss = loss + weight * reg

    return loss
