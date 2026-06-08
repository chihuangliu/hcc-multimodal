"""Image and gene encoders for contrastive learning."""

import torch
import torch.nn as nn
from torchvision.models import ViT_B_16_Weights, ViT_B_32_Weights, vit_b_16, vit_b_32
from torchvision.transforms import v2


class _HFViTWrapper(nn.Module):
    """Wraps a HuggingFace ViT to return the CLS token embedding."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).last_hidden_state[:, 0]


def _load_vit_b_32():
    m = vit_b_32(weights=ViT_B_32_Weights.IMAGENET1K_V1)
    m.heads = nn.Identity()
    return m, 768


def _load_vit_b_16():
    m = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    m.heads = nn.Identity()
    return m, 768


def _load_dinov2_vitb14():
    from transformers import AutoModel
    m = AutoModel.from_pretrained("facebook/dinov2-base")
    return _HFViTWrapper(m), 768


def _load_dinov3_vitb16():
    from transformers import AutoModel
    m = AutoModel.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m")
    return _HFViTWrapper(m), 768


def _imagenet_transforms():
    return v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# Each entry: backbone_name -> loader() returns (nn.Module, feat_dim)
BACKBONES = {
    "vit_b_32": _load_vit_b_32,
    "vit_b_16": _load_vit_b_16,
    "dinov2_vitb14": _load_dinov2_vitb14,
    "dinov3_vitb16": _load_dinov3_vitb16,
}

# Normalisation transforms for each backbone (all use ImageNet stats)
BACKBONE_TRANSFORMS = {
    "vit_b_32": ViT_B_32_Weights.IMAGENET1K_V1.transforms,
    "vit_b_16": ViT_B_16_Weights.IMAGENET1K_V1.transforms,
    "dinov2_vitb14": _imagenet_transforms,
    "dinov3_vitb16": _imagenet_transforms,
}


class ImageEncoder(nn.Module):
    def __init__(self, backbone_name: str, embed_dim: int, freeze: bool):
        super().__init__()
        backbone, feat_dim = BACKBONES[backbone_name]()
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
