"""Image and gene encoders for contrastive learning."""

import torch
import torch.nn as nn
from torchvision.models import ViT_B_16_Weights, ViT_B_32_Weights, vit_b_16, vit_b_32
from torchvision.transforms import v2

# HuggingFace ViT backbone registry: name -> (model_id, feat_dim)
_HF_BACKBONE_CONFIGS = {
    "dinov2_vits14": ("facebook/dinov2-small", 384),
    "dinov2_vitb14": ("facebook/dinov2-base", 768),
    "dinov2_vitl14": ("facebook/dinov2-large", 1024),
    "dinov3_vits16": ("facebook/dinov3-vits16-pretrain-lvd1689m", 384),
    "dinov3_vitb16": ("facebook/dinov3-vitb16-pretrain-lvd1689m", 768),
    "dinov3_vitl16": ("facebook/dinov3-vitl16-pretrain-lvd1689m", 1024),
}


class _HFViTWrapper(nn.Module):
    """Wraps a HuggingFace ViT to return the CLS token embedding."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).last_hidden_state[:, 0]


def _make_hf_loader(model_id: str, feat_dim: int):
    def load():
        from transformers import AutoModel
        return _HFViTWrapper(AutoModel.from_pretrained(model_id)), feat_dim
    return load


def _load_vit_b_32():
    m = vit_b_32(weights=ViT_B_32_Weights.IMAGENET1K_V1)
    m.heads = nn.Identity()
    return m, 768


def _load_vit_b_16():
    m = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    m.heads = nn.Identity()
    return m, 768


def _imagenet_transforms():
    return v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


BACKBONES = {
    "vit_b_32": _load_vit_b_32,
    "vit_b_16": _load_vit_b_16,
    **{name: _make_hf_loader(mid, dim) for name, (mid, dim) in _HF_BACKBONE_CONFIGS.items()},
}

BACKBONE_TRANSFORMS = {
    "vit_b_32": ViT_B_32_Weights.IMAGENET1K_V1.transforms,
    "vit_b_16": ViT_B_16_Weights.IMAGENET1K_V1.transforms,
    **{name: _imagenet_transforms for name in _HF_BACKBONE_CONFIGS},
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
