"""Image and gene encoders for contrastive learning."""

import torch
import torch.nn as nn
from torchvision.models import ViT_B_32_Weights, vit_b_32

BACKBONES = {
    "vit_b_32": (vit_b_32, ViT_B_32_Weights.IMAGENET1K_V1, 768),
}


class ImageEncoder(nn.Module):
    def __init__(self, backbone_name: str, embed_dim: int, freeze: bool):
        super().__init__()
        factory, weights, feat_dim = BACKBONES[backbone_name]
        backbone = factory(weights=weights)
        backbone.heads = nn.Identity()
        self.backbone = backbone
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.backbone(x))


class GeneEncoder(nn.Module):
    def __init__(self, gene_dim: int, hidden_dim: int, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(gene_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
