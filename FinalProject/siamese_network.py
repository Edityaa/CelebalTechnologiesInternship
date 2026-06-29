"""
models/siamese_network.py
--------------------------
Siamese network with triplet loss for learning a semantic embedding space.

Architecture:
  - Shared FeatureExtractor encoder (ResNet50 + projection head)
  - Triplet loss with hard negative mining within each batch

Dataset:
  - TripletFashionDataset: samples (anchor, positive, negative) triplets
    by grouping images by category from the manifest CSV.
"""

from __future__ import annotations

import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd

from models.feature_extractor import FeatureExtractor
from utils.preprocessing import get_train_transform, get_inference_transform


# ── Dataset ──────────────────────────────────────────────────────────────────

class TripletFashionDataset(Dataset):
    """
    Returns (anchor, positive, negative) image triplets.

    Sampling strategy:
      - Anchor:   randomly chosen image from any category.
      - Positive: different image from the SAME category.
      - Negative: image from a DIFFERENT category (semi-hard: random for now).

    Args:
        manifest_path:  Path to manifest.csv (from subset_builder).
        split:          "train" | "val".
        transform:      torchvision transform applied to every image.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split: str = "train",
        transform=None,
    ):
        df = pd.read_csv(manifest_path)
        df = df[df["split"] == split].reset_index(drop=True)

        self.transform = transform or get_train_transform()
        self.samples = df[["image_path", "category"]].values.tolist()

        # group indices by category for fast positive/negative sampling
        self.cat_to_indices: dict[str, list[int]] = {}
        for idx, (_, cat) in enumerate(self.samples):
            self.cat_to_indices.setdefault(cat, []).append(idx)

        self.categories = list(self.cat_to_indices.keys())

    def __len__(self) -> int:
        return len(self.samples)

    def _load(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        return self.transform(img)

    def __getitem__(self, idx: int):
        anchor_path, anchor_cat = self.samples[idx]

        # Positive: same category, different index
        pos_pool = [i for i in self.cat_to_indices[anchor_cat] if i != idx]
        pos_idx = random.choice(pos_pool) if pos_pool else idx
        pos_path, _ = self.samples[pos_idx]

        # Negative: different category
        neg_cat = random.choice([c for c in self.categories if c != anchor_cat])
        neg_idx = random.choice(self.cat_to_indices[neg_cat])
        neg_path, _ = self.samples[neg_idx]

        return (
            self._load(anchor_path),
            self._load(pos_path),
            self._load(neg_path),
            anchor_cat,
        )


# ── Loss ─────────────────────────────────────────────────────────────────────

class TripletLoss(nn.Module):
    """
    Standard triplet loss:
        L = max(0, d(a, p) - d(a, n) + margin)

    where d(·, ·) is squared L2 distance on unit-norm embeddings
    (equivalent to 2 - 2·cosine_similarity).

    Args:
        margin: Separation margin (default 0.5).
    """

    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        d_pos = F.pairwise_distance(anchor, positive, p=2)
        d_neg = F.pairwise_distance(anchor, negative, p=2)
        losses = F.relu(d_pos - d_neg + self.margin)
        return losses.mean()


# ── Siamese network wrapper ───────────────────────────────────────────────────

class SiameseNetwork(nn.Module):
    """
    Thin wrapper: the encoder is a shared FeatureExtractor.
    Forward accepts three image batches (anchor, positive, negative)
    and returns their embeddings.

    Args:
        backbone:       Backbone name forwarded to FeatureExtractor.
        embedding_dim:  Output embedding dimension.
        margin:         Triplet loss margin.
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        embedding_dim: int = 128,
        margin: float = 0.5,
    ):
        super().__init__()
        self.encoder = FeatureExtractor.for_siamese(
            backbone=backbone, embedding_dim=embedding_dim
        )
        self.criterion = TripletLoss(margin=margin)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience: encode a single batch."""
        return self.encoder(x)

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            loss:       Scalar triplet loss.
            embeddings: Tuple of (anchor_emb, pos_emb, neg_emb).
        """
        a_emb = self.encoder(anchor)
        p_emb = self.encoder(positive)
        n_emb = self.encoder(negative)
        loss = self.criterion(a_emb, p_emb, n_emb)
        return loss, (a_emb, p_emb, n_emb)

    def save(self, path: str | Path):
        torch.save(self.state_dict(), path)
        print(f"Model saved → {path}")

    @classmethod
    def load(
        cls,
        path: str | Path,
        backbone: str = "resnet50",
        embedding_dim: int = 128,
        device: str = "cpu",
    ) -> "SiameseNetwork":
        model = cls(backbone=backbone, embedding_dim=embedding_dim)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        return model
