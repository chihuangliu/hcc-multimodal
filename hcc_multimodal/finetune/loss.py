"""DINO self-distillation loss and projection head."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOHead(nn.Module):
    """MLP projection head with L2-normalised bottleneck and unit-norm last layer.

    Architecture: Linear → BN → GELU → Linear → BN → GELU → Linear(bottleneck)
                  → L2-norm → Linear(out_dim, no bias, weights unit-normalised in forward)
    """

    def __init__(self, in_dim: int, out_dim: int = 4096, hidden_dim: int = 2048, bottleneck_dim: int = 256):
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


class DINOLoss(nn.Module):
    """Cross-entropy loss between teacher and student softmax distributions.

    The teacher output is centered (running mean subtracted) and sharpened
    (low temperature) to produce a peaked target distribution. The student
    output uses a higher temperature for a softer distribution.

    Centering prevents mode collapse without requiring contrastive negative
    pairs.

    Args:
        out_dim: prototype/softmax dimension K
        student_temp: student softmax temperature (default 0.1)
        center_momentum: EMA momentum for center update (default 0.9)
    """

    def __init__(
        self,
        out_dim: int,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(
        self,
        student_output: list[torch.Tensor],
        teacher_output: list[torch.Tensor],
        teacher_temp: float,
    ) -> torch.Tensor:
        """Compute DINO loss across all student/teacher view pairs.

        Args:
            student_output: list of (N, K) logit tensors — all crops (global + local)
            teacher_output: list of (N, K) logit tensors — global crops only
            teacher_temp: current teacher temperature (warmed up externally)

        Returns:
            Scalar mean loss across all student-teacher pairs (excluding same-view pairs).
        """
        teacher_probs = [
            F.softmax((t - self.center) / teacher_temp, dim=-1).detach()
            for t in teacher_output
        ]
        student_log_probs = [
            F.log_softmax(s / self.student_temp, dim=-1)
            for s in student_output
        ]

        n_global = len(teacher_output)
        total_loss = torch.tensor(0.0, device=student_output[0].device)
        n_terms = 0

        for i, p_t in enumerate(teacher_probs):
            for j, log_p_s in enumerate(student_log_probs):
                if i == j:
                    continue
                total_loss += -(p_t * log_p_s).sum(dim=-1).mean()
                n_terms += 1

        return total_loss / n_terms

    @torch.no_grad()
    def update_center(self, teacher_output: list[torch.Tensor]) -> None:
        """EMA update of the centering buffer from teacher global-crop outputs."""
        batch_center = torch.cat(teacher_output, dim=0).mean(dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
