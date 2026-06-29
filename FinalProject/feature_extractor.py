"""
models/feature_extractor.py
----------------------------
Wraps a pretrained CNN backbone (ResNet50 or EfficientNet-B0) to produce
L2-normalised embeddings of configurable dimension.

The classification head is replaced with:
    GlobalAvgPool → Dropout → Linear(backbone_dim → embedding_dim) → L2 Norm

This module is used as the shared encoder inside the Siamese network and
also for the baseline (frozen) feature extraction.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


class FeatureExtractor(nn.Module):
    """
    Pretrained CNN backbone with a custom projection head.

    Args:
        backbone:       "resnet50" | "efficientnet_b0"
        embedding_dim:  Output embedding size (default 128).
        freeze_base:    If True, only the projection head is trainable.
                        Set False when fine-tuning last CNN layers.
        dropout:        Dropout probability before the projection layer.
    """

    SUPPORTED = ("resnet50", "efficientnet_b0")

    def __init__(
        self,
        backbone: str = "resnet50",
        embedding_dim: int = 128,
        freeze_base: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()

        if backbone not in self.SUPPORTED:
            raise ValueError(f"backbone must be one of {self.SUPPORTED}, got '{backbone}'")

        self.backbone_name = backbone
        self.embedding_dim = embedding_dim

        # ── load pretrained backbone ─────────────────────────────────────────
        if backbone == "resnet50":
            base = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V2)
            feature_dim = base.fc.in_features          # 2048
            base.fc = nn.Identity()                    # remove classifier head
        else:  # efficientnet_b0
            base = tv_models.efficientnet_b0(weights=tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            feature_dim = base.classifier[1].in_features  # 1280
            base.classifier = nn.Identity()

        self.base = base

        # ── optionally freeze the backbone ───────────────────────────────────
        if freeze_base:
            for param in self.base.parameters():
                param.requires_grad = False
        else:
            # Freeze only the first ~70 % of layers; fine-tune the rest
            self._partial_freeze(freeze_ratio=0.7)

        # ── projection head ──────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, embedding_dim),
        )

    # ── helpers ──────────────────────────────────────────────────────────────

    def _partial_freeze(self, freeze_ratio: float = 0.7):
        """Freeze the first `freeze_ratio` fraction of backbone parameters."""
        params = list(self.base.parameters())
        cutoff = int(len(params) * freeze_ratio)
        for p in params[:cutoff]:
            p.requires_grad = False
        for p in params[cutoff:]:
            p.requires_grad = True

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 224, 224) image batch.

        Returns:
            embeddings: (B, embedding_dim) L2-normalised vectors.
        """
        features = self.base(x)          # (B, feature_dim)
        projected = self.head(features)  # (B, embedding_dim)
        return F.normalize(projected, p=2, dim=1)

    # ── convenience constructors ─────────────────────────────────────────────

    @classmethod
    def baseline(cls, backbone: str = "resnet50", embedding_dim: int = 128) -> "FeatureExtractor":
        """Fully frozen backbone — used for zero-shot baseline comparisons."""
        return cls(backbone=backbone, embedding_dim=embedding_dim, freeze_base=True)

    @classmethod
    def for_siamese(cls, backbone: str = "resnet50", embedding_dim: int = 128) -> "FeatureExtractor":
        """Partially unfrozen backbone — used as the shared encoder in Siamese training."""
        return cls(backbone=backbone, embedding_dim=embedding_dim, freeze_base=False)
