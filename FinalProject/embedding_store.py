"""
embeddings/embedding_store.py
------------------------------
Precomputes embeddings for every image in the subset, builds a FAISS index
for fast cosine similarity search, and provides a simple query interface.

Why FAISS IndexFlatIP?
  - Embeddings are L2-normalised → inner product == cosine similarity.
  - IndexFlatIP gives exact (brute-force) search; sufficient for <10 k images.
  - For datasets >100 k, swap in IndexIVFFlat with nlist ≈ sqrt(N).

Usage (programmatic):
    store = EmbeddingStore.load("embeddings/faiss_index.bin", "embeddings/metadata.pkl")
    results = store.query(query_image_tensor, k=5)
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from utils.preprocessing import get_inference_transform


class EmbeddingStore:
    """
    Manages the FAISS index and associated image metadata.

    Attributes:
        index:    FAISS IndexFlatIP (cosine similarity via inner product on L2-norm vectors).
        metadata: List of dicts with keys {image_path, category, id}.
        transform: Inference-time image transform.
    """

    def __init__(self):
        self.index: Optional[faiss.IndexFlatIP] = None
        self.metadata: list[dict] = []
        self.transform = get_inference_transform()

    # ── building ──────────────────────────────────────────────────────────────

    def build(
        self,
        manifest_path: str | Path,
        encoder,                        # FeatureExtractor or SiameseNetwork.encoder
        device: torch.device,
        batch_size: int = 64,
    ) -> None:
        """
        Encodes all images in the manifest and populates the FAISS index.

        Args:
            manifest_path:  Path to manifest.csv.
            encoder:        Callable that maps (B, 3, H, W) → (B, D) embeddings.
            device:         Torch device.
            batch_size:     Encoding batch size.
        """
        df = pd.read_csv(manifest_path)
        paths      = df["image_path"].tolist()
        categories = df["category"].tolist()
        ids        = df["id"].tolist()

        encoder.eval()
        all_embeddings = []

        with torch.no_grad():
            for start in tqdm(
                range(0, len(paths), batch_size),
                desc="Building embeddings",
                unit="batch",
            ):
                batch_paths = paths[start : start + batch_size]
                tensors = []
                valid_indices = []

                for i, p in enumerate(batch_paths):
                    try:
                        img = Image.open(p).convert("RGB")
                        tensors.append(self.transform(img))
                        valid_indices.append(start + i)
                    except Exception as e:
                        print(f"  ⚠ Skipping {p}: {e}")

                if not tensors:
                    continue

                batch = torch.stack(tensors).to(device)
                embs  = encoder(batch).cpu().numpy().astype("float32")
                all_embeddings.append(embs)

                for i in valid_indices:
                    self.metadata.append({
                        "image_path": paths[i],
                        "category":   categories[i],
                        "id":         ids[i],
                    })

        if not all_embeddings:
            raise RuntimeError("No embeddings produced; check image paths in manifest.")

        matrix = np.vstack(all_embeddings)
        # Embeddings are already L2-normalised by FeatureExtractor
        # Normalise again defensively
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = matrix / np.clip(norms, 1e-8, None)

        dim = matrix.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # inner product == cosine on unit vectors
        self.index.add(matrix)

        print(f"✅ Index built: {self.index.ntotal} vectors of dim {dim}")

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self, index_path: str | Path, meta_path: str | Path) -> None:
        index_path = Path(index_path)
        meta_path  = Path(meta_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"Index saved → {index_path}")
        print(f"Metadata saved → {meta_path}")

    @classmethod
    def load(cls, index_path: str | Path, meta_path: str | Path) -> "EmbeddingStore":
        store = cls()
        store.index = faiss.read_index(str(index_path))
        with open(meta_path, "rb") as f:
            store.metadata = pickle.load(f)
        print(f"Loaded index ({store.index.ntotal} vectors) from {index_path}")
        return store

    # ── querying ──────────────────────────────────────────────────────────────

    def query(
        self,
        query_tensor: torch.Tensor,     # (1, 3, H, W) or (3, H, W)
        encoder,
        device: torch.device,
        k: int = 5,
    ) -> list[dict]:
        """
        Find the top-K most similar images to a query.

        Args:
            query_tensor:  Pre-processed image tensor (single image).
            encoder:       Same encoder used to build the index.
            device:        Torch device.
            k:             Number of results to return.

        Returns:
            List of dicts with keys: image_path, category, similarity_score, rank.
        """
        if query_tensor.dim() == 3:
            query_tensor = query_tensor.unsqueeze(0)

        encoder.eval()
        t0 = time.perf_counter()

        with torch.no_grad():
            emb = encoder(query_tensor.to(device)).cpu().numpy().astype("float32")

        # Normalise query vector
        norm = np.linalg.norm(emb, axis=1, keepdims=True)
        emb  = emb / np.clip(norm, 1e-8, None)

        scores, indices = self.index.search(emb, k + 1)  # +1 to skip self if present
        elapsed_ms = (time.perf_counter() - t0) * 1000

        results = []
        seen_paths = set()
        # Get query path if it's in the index (skip it)
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            meta = self.metadata[idx]
            if meta["image_path"] in seen_paths:
                continue
            seen_paths.add(meta["image_path"])
            results.append({
                "image_path":       meta["image_path"],
                "category":         meta["category"],
                "id":               meta["id"],
                "similarity_score": float(score),
                "rank":             len(results) + 1,
            })
            if len(results) >= k:
                break

        return results, elapsed_ms

    def query_from_pil(
        self,
        pil_image: Image.Image,
        encoder,
        device: torch.device,
        k: int = 5,
    ) -> tuple[list[dict], float]:
        """Convenience: query directly from a PIL Image."""
        tensor = self.transform(pil_image.convert("RGB"))
        return self.query(tensor, encoder, device, k)
