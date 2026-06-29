"""
utils/evaluation.py
--------------------
Quantitative evaluation of the recommendation system.

Metrics:
  - Precision@K: fraction of top-K results in the same category as the query.
  - Recall@K:    fraction of same-category images retrieved in top-K.
  - Mean inference time (ms) over N queries.

Usage:
    python utils/evaluation.py \
        --data_dir    data/subset \
        --siamese_model models/siamese_checkpoint.pt \
        --k 5 \
        --n_queries 200
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from embeddings.embedding_store import EmbeddingStore
from models.feature_extractor import FeatureExtractor
from models.siamese_network import SiameseNetwork
from utils.preprocessing import get_inference_transform


# ── metric functions ──────────────────────────────────────────────────────────

def precision_at_k(results: list[dict], query_category: str, k: int) -> float:
    """Fraction of top-K results in the same category."""
    top_k = results[:k]
    if not top_k:
        return 0.0
    relevant = sum(1 for r in top_k if r["category"] == query_category)
    return relevant / len(top_k)


def recall_at_k(
    results: list[dict],
    query_category: str,
    total_relevant: int,
    k: int,
) -> float:
    """Fraction of all relevant items that appear in top-K."""
    if total_relevant == 0:
        return 0.0
    top_k = results[:k]
    retrieved = sum(1 for r in top_k if r["category"] == query_category)
    return retrieved / total_relevant


# ── evaluation runner ────────────────────────────────────────────────────────

def evaluate_system(
    store: EmbeddingStore,
    encoder,
    manifest: pd.DataFrame,
    device: torch.device,
    k: int = 5,
    n_queries: int = 200,
    seed: int = 42,
) -> dict:
    """
    Runs Precision@K, Recall@K, and timing over a random sample of queries.

    Returns a dict with keys: precision_at_k, recall_at_k, mean_inference_ms.
    """
    random.seed(seed)
    transform = get_inference_transform()

    query_rows = manifest.sample(min(n_queries, len(manifest)), random_state=seed)

    # count total relevant per category in the index
    cat_counts: dict[str, int] = manifest["category"].value_counts().to_dict()

    precisions = []
    recalls    = []
    times_ms   = []

    for _, row in tqdm(query_rows.iterrows(), total=len(query_rows), desc="Evaluating"):
        try:
            img = Image.open(row["image_path"]).convert("RGB")
        except Exception:
            continue

        tensor = transform(img)
        results, elapsed_ms = store.query(tensor, encoder, device, k=k)

        cat = row["category"]
        p = precision_at_k(results, cat, k)
        r = recall_at_k(results, cat, cat_counts.get(cat, 1) - 1, k)

        precisions.append(p)
        recalls.append(r)
        times_ms.append(elapsed_ms)

    return {
        f"precision@{k}":       np.mean(precisions),
        f"recall@{k}":          np.mean(recalls),
        "mean_inference_ms":    np.mean(times_ms),
        "std_inference_ms":     np.std(times_ms),
        "n_queries":            len(precisions),
    }


def compare_models(
    manifest_path: str | Path,
    siamese_model_path: str | None,
    baseline_index_path: str,
    baseline_meta_path: str,
    siamese_index_path: str | None,
    siamese_meta_path: str | None,
    k: int = 5,
    n_queries: int = 200,
    backbone: str = "resnet50",
    embedding_dim: int = 128,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = pd.read_csv(manifest_path)

    systems = {}

    # ── baseline ──────────────────────────────────────────────────────────────
    baseline_enc = FeatureExtractor.baseline(backbone, embedding_dim).to(device)
    baseline_store = EmbeddingStore.load(baseline_index_path, baseline_meta_path)
    systems["Baseline (frozen ResNet50)"] = (baseline_store, baseline_enc)

    # ── siamese ───────────────────────────────────────────────────────────────
    if siamese_model_path and siamese_index_path:
        siamese = SiameseNetwork.load(
            siamese_model_path, backbone=backbone,
            embedding_dim=embedding_dim, device=str(device)
        )
        siamese_enc = siamese.encoder.to(device)
        siamese_store = EmbeddingStore.load(siamese_index_path, siamese_meta_path)
        systems["Siamese (fine-tuned)"] = (siamese_store, siamese_enc)

    # ── run & display ─────────────────────────────────────────────────────────
    print(f"\n{'System':<30} {'P@'+str(k):>8} {'R@'+str(k):>8} {'Latency (ms)':>14}")
    print("─" * 66)

    for name, (store, encoder) in systems.items():
        metrics = evaluate_system(store, encoder, manifest, device, k=k, n_queries=n_queries)
        p  = metrics[f"precision@{k}"]
        r  = metrics[f"recall@{k}"]
        ms = metrics["mean_inference_ms"]
        print(f"{name:<30} {p:>8.3f} {r:>8.3f} {ms:>14.2f}")

    print()


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",            default="data/subset")
    parser.add_argument("--siamese_model",        default=None)
    parser.add_argument("--baseline_index",       default="embeddings/baseline_index.bin")
    parser.add_argument("--baseline_meta",        default="embeddings/baseline_meta.pkl")
    parser.add_argument("--siamese_index",        default="embeddings/siamese_index.bin")
    parser.add_argument("--siamese_meta",         default="embeddings/siamese_meta.pkl")
    parser.add_argument("--k",                   type=int, default=5)
    parser.add_argument("--n_queries",           type=int, default=200)
    parser.add_argument("--backbone",            default="resnet50")
    parser.add_argument("--embedding_dim",       type=int, default=128)
    args = parser.parse_args()

    compare_models(
        manifest_path       = Path(args.data_dir) / "manifest.csv",
        siamese_model_path  = args.siamese_model,
        baseline_index_path = args.baseline_index,
        baseline_meta_path  = args.baseline_meta,
        siamese_index_path  = args.siamese_index if args.siamese_model else None,
        siamese_meta_path   = args.siamese_meta  if args.siamese_model else None,
        k                   = args.k,
        n_queries           = args.n_queries,
        backbone            = args.backbone,
        embedding_dim       = args.embedding_dim,
    )


if __name__ == "__main__":
    main()
