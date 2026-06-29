"""
build_index.py
--------------
Encodes all images in the subset and writes a FAISS index to disk.

Usage:
    # Baseline (frozen ResNet50)
    python build_index.py \
        --data_dir    data/subset \
        --index_path  embeddings/baseline_index.bin \
        --meta_path   embeddings/baseline_meta.pkl

    # Siamese-trained model
    python build_index.py \
        --data_dir    data/subset \
        --model_path  models/siamese_checkpoint.pt \
        --index_path  embeddings/siamese_index.bin \
        --meta_path   embeddings/siamese_meta.pkl
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from embeddings.embedding_store import EmbeddingStore
from models.feature_extractor import FeatureExtractor
from models.siamese_network import SiameseNetwork


def build(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── load encoder ─────────────────────────────────────────────────────────
    if args.model_path:
        print(f"Loading Siamese encoder from {args.model_path} ...")
        siamese = SiameseNetwork.load(
            args.model_path,
            backbone=args.backbone,
            embedding_dim=args.embedding_dim,
            device=str(device),
        )
        encoder = siamese.encoder.to(device)
    else:
        print("Using baseline (frozen ResNet50) encoder ...")
        encoder = FeatureExtractor.baseline(
            backbone=args.backbone,
            embedding_dim=args.embedding_dim,
        ).to(device)

    # ── build & save index ────────────────────────────────────────────────────
    manifest = Path(args.data_dir) / "manifest.csv"
    store = EmbeddingStore()
    store.build(
        manifest_path=manifest,
        encoder=encoder,
        device=device,
        batch_size=args.batch_size,
    )
    store.save(args.index_path, args.meta_path)
    print("Done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",      default="data/subset")
    parser.add_argument("--model_path",    default=None,
                        help="Path to Siamese checkpoint (omit for baseline)")
    parser.add_argument("--backbone",      default="resnet50",
                        choices=["resnet50", "efficientnet_b0"])
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--batch_size",    type=int, default=64)
    parser.add_argument("--index_path",    default="embeddings/faiss_index.bin")
    parser.add_argument("--meta_path",     default="embeddings/metadata.pkl")
    args = parser.parse_args()

    build(args)


if __name__ == "__main__":
    main()
