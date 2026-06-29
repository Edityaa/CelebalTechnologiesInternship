"""
train_siamese.py
-----------------
End-to-end training script for the Siamese network.

Runs:
  1. Build train / val DataLoaders from the manifest CSV.
  2. Train with AdamW + cosine LR schedule.
  3. Log train/val loss every epoch; save best checkpoint.
  4. Print a summary table on exit.

Usage:
    python train_siamese.py \
        --data_dir data/subset \
        --backbone resnet50 \
        --epochs 20 \
        --batch_size 32 \
        --embedding_dim 128 \
        --margin 0.5 \
        --lr 1e-4 \
        --save_path models/siamese_checkpoint.pt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from models.siamese_network import SiameseNetwork, TripletFashionDataset
from utils.preprocessing import get_train_transform, get_inference_transform


def train_one_epoch(
    model: SiameseNetwork,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for anchor, positive, negative, _ in loader:
        anchor   = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        optimizer.zero_grad()
        loss, _ = model(anchor, positive, negative)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model: SiameseNetwork,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    for anchor, positive, negative, _ in loader:
        anchor   = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)
        loss, _ = model(anchor, positive, negative)
        total_loss += loss.item()
    return total_loss / len(loader)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    manifest = Path(args.data_dir) / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(
            f"manifest.csv not found at {manifest}. "
            "Run data/subset_builder.py first."
        )

    # ── datasets & loaders ───────────────────────────────────────────────────
    train_ds = TripletFashionDataset(manifest, split="train", transform=get_train_transform())
    val_ds   = TripletFashionDataset(manifest, split="val",   transform=get_inference_transform())

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    print(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")

    # ── model ────────────────────────────────────────────────────────────────
    model = SiameseNetwork(
        backbone=args.backbone,
        embedding_dim=args.embedding_dim,
        margin=args.margin,
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {trainable:,} trainable / {total:,} total")

    # ── optimiser & scheduler ────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # ── training loop ────────────────────────────────────────────────────────
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    history  = []

    print(f"\n{'Epoch':>6} {'Train Loss':>12} {'Val Loss':>10} {'Time (s)':>10}")
    print("─" * 44)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss   = evaluate(model, val_loader, device)
        scheduler.step()
        elapsed = time.time() - t0

        history.append({"epoch": epoch, "train": train_loss, "val": val_loss})
        print(f"{epoch:>6} {train_loss:>12.4f} {val_loss:>10.4f} {elapsed:>10.1f}")

        if val_loss < best_val:
            best_val = val_loss
            model.save(save_path)
            print(f"         ↑ saved checkpoint (val={best_val:.4f})")

    print(f"\nTraining complete. Best val loss: {best_val:.4f}")
    print(f"Checkpoint: {save_path}")
    return history


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train the Siamese recommendation model")
    parser.add_argument("--data_dir",      default="data/subset")
    parser.add_argument("--backbone",      default="resnet50", choices=["resnet50", "efficientnet_b0"])
    parser.add_argument("--epochs",        type=int,   default=20)
    parser.add_argument("--batch_size",    type=int,   default=32)
    parser.add_argument("--embedding_dim", type=int,   default=128)
    parser.add_argument("--margin",        type=float, default=0.5)
    parser.add_argument("--lr",            type=float, default=1e-4)
    parser.add_argument("--save_path",     default="models/siamese_checkpoint.pt")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
