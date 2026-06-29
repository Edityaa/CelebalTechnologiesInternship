"""
app/streamlit_app.py
---------------------
Interactive Streamlit UI for the Visual Product Recommendation System.

Features:
  - Upload any product image
  - Choose between Baseline and Siamese-trained model
  - Set K (number of results)
  - Displays: query image, top-K results grid, similarity scores, inference time
  - Side-by-side model comparison mode

Run:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# ensure project root is on the path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import streamlit as st
import torch
from PIL import Image

# ── page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Visual Product Search",
    page_icon="👗",
    layout="wide",
)

BASELINE_INDEX  = ROOT / "embeddings" / "baseline_index.bin"
BASELINE_META   = ROOT / "embeddings" / "baseline_meta.pkl"
SIAMESE_INDEX   = ROOT / "embeddings" / "siamese_index.bin"
SIAMESE_META    = ROOT / "embeddings" / "siamese_meta.pkl"
SIAMESE_CKPT    = ROOT / "models"     / "siamese_checkpoint.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── cached model loaders ──────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading baseline model …")
def load_baseline():
    from embeddings.embedding_store import EmbeddingStore
    from models.feature_extractor import FeatureExtractor

    encoder = FeatureExtractor.baseline(backbone="resnet50", embedding_dim=128).to(DEVICE)
    store   = EmbeddingStore.load(BASELINE_INDEX, BASELINE_META)
    return store, encoder


@st.cache_resource(show_spinner="Loading Siamese model …")
def load_siamese():
    from embeddings.embedding_store import EmbeddingStore
    from models.siamese_network import SiameseNetwork

    siamese = SiameseNetwork.load(
        SIAMESE_CKPT, backbone="resnet50", embedding_dim=128, device=str(DEVICE)
    )
    encoder = siamese.encoder.to(DEVICE)
    store   = EmbeddingStore.load(SIAMESE_INDEX, SIAMESE_META)
    return store, encoder


# ── helpers ───────────────────────────────────────────────────────────────────

def results_grid(results: list[dict], scores: list[float], cols_per_row: int = 5):
    """Render a grid of result images with category and score captions."""
    cols = st.columns(cols_per_row)
    for i, (res, score) in enumerate(zip(results, scores)):
        col = cols[i % cols_per_row]
        try:
            img = Image.open(res["image_path"]).convert("RGB")
            col.image(
                img,
                caption=f"#{res['rank']} {res['category']}\nScore: {score:.3f}",
                use_column_width=True,
            )
        except Exception:
            col.warning(f"Image not found:\n{res['image_path']}")


def similarity_bar(results: list[dict]):
    """Show cosine similarity as a horizontal bar chart."""
    import pandas as pd

    data = {
        "Label": [f"#{r['rank']} {r['category']}" for r in results],
        "Cosine Similarity": [r["similarity_score"] for r in results],
    }
    df = pd.DataFrame(data).set_index("Label")
    st.bar_chart(df)


# ── main app ──────────────────────────────────────────────────────────────────

def main():
    st.title("👗 Visual Product Recommendation")
    st.caption(
        "Upload a product image and retrieve visually similar items "
        "using deep learning embeddings and cosine similarity."
    )

    # ── sidebar controls ──────────────────────────────────────────────────────
    st.sidebar.header("⚙️ Settings")

    model_choice = st.sidebar.radio(
        "Model",
        options=["Baseline (ResNet50)", "Siamese (fine-tuned)", "Compare both"],
        index=1,
    )

    k = st.sidebar.slider("Top-K results", min_value=1, max_value=20, value=6)

    show_scores = st.sidebar.checkbox("Show similarity chart", value=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Dataset:** [Fashion Product Images](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset)  \n"
        "**Backbone:** ResNet50 (ImageNet)  \n"
        "**Loss:** Triplet (margin=0.5)  \n"
        "**Search:** FAISS IndexFlatIP"
    )

    # ── check index availability ──────────────────────────────────────────────
    baseline_ready = BASELINE_INDEX.exists() and BASELINE_META.exists()
    siamese_ready  = SIAMESE_INDEX.exists()  and SIAMESE_META.exists() and SIAMESE_CKPT.exists()

    if not baseline_ready and not siamese_ready:
        st.error(
            "⚠️ No FAISS index found. Please run the setup pipeline:\n"
            "```\n"
            "python data/subset_builder.py --dataset_dir data/fashion-dataset --output_dir data/subset\n"
            "python train_siamese.py\n"
            "python build_index.py\n"
            "```"
        )
        return

    # ── image upload ──────────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Upload a product image (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded is None:
        st.info("👆 Upload an image to see similar products.")
        return

    query_image = Image.open(uploaded).convert("RGB")

    col_query, col_spacer = st.columns([1, 3])
    with col_query:
        st.image(query_image, caption="Query image", use_column_width=True)

    # ── run search ────────────────────────────────────────────────────────────

    def run_query(store, encoder, label: str):
        results, elapsed_ms = store.query_from_pil(query_image, encoder, DEVICE, k=k)
        st.subheader(f"🔍 {label}")
        st.caption(f"Inference time: **{elapsed_ms:.1f} ms**")
        results_grid(results, [r["similarity_score"] for r in results])
        if show_scores:
            with st.expander("Similarity scores"):
                similarity_bar(results)
        return results, elapsed_ms

    use_baseline = model_choice in ("Baseline (ResNet50)", "Compare both")
    use_siamese  = model_choice in ("Siamese (fine-tuned)", "Compare both")

    if use_baseline:
        if not baseline_ready:
            st.warning("Baseline index not found. Run `build_index.py` without `--model_path`.")
        else:
            b_store, b_enc = load_baseline()
            run_query(b_store, b_enc, "Baseline — Frozen ResNet50 Embeddings")

    if use_siamese:
        if not siamese_ready:
            st.warning("Siamese index not found. Run `train_siamese.py` then `build_index.py`.")
        else:
            s_store, s_enc = load_siamese()
            run_query(s_store, s_enc, "Siamese — Fine-tuned with Triplet Loss")


if __name__ == "__main__":
    main()
