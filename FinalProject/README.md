# Visual Product Recommendation System

An image-based recommendation engine using deep learning embeddings, transfer learning, and Siamese networks trained on the [Fashion Product Images dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset).

---

## Project Structure

```
visual_product_rec/
├── data/
│   └── subset_builder.py       # Downloads & builds the training subset
├── models/
│   ├── feature_extractor.py    # Pretrained CNN backbone (ResNet50/EfficientNet)
│   └── siamese_network.py      # Siamese network with triplet loss
├── embeddings/
│   └── embedding_store.py      # Precompute & persist embeddings + FAISS index
├── utils/
│   ├── preprocessing.py        # Image transforms & augmentation
│   └── evaluation.py           # Precision@K, Recall@K, inference timing
├── app/
│   └── streamlit_app.py        # Interactive Streamlit UI
├── train_siamese.py            # Full training pipeline
├── build_index.py              # Build / refresh the FAISS embedding index
└── requirements.txt
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download the dataset
Get the dataset from Kaggle:
```
https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset
```
Unzip and place at `data/fashion-dataset/` so the path looks like:
```
data/fashion-dataset/images/
data/fashion-dataset/styles.csv
```

### 3. Build the training subset
```bash
python data/subset_builder.py \
    --dataset_dir data/fashion-dataset \
    --output_dir data/subset \
    --categories "Shirts,Shoes,Dresses,Watches,Bags,Sunglasses,Jeans" \
    --samples_per_category 250
```

### 4. Train the Siamese Network
```bash
python train_siamese.py \
    --data_dir data/subset \
    --backbone resnet50 \
    --epochs 20 \
    --batch_size 32 \
    --embedding_dim 128 \
    --save_path models/siamese_checkpoint.pt
```

### 5. Build the FAISS index
```bash
python build_index.py \
    --data_dir data/subset \
    --model_path models/siamese_checkpoint.pt \
    --index_path embeddings/faiss_index.bin \
    --meta_path embeddings/metadata.pkl
```

### 6. Launch the UI
```bash
streamlit run app/streamlit_app.py
```

---

## Key Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Backbone | ResNet50 | Strong ImageNet features, widely supported |
| Loss | Triplet loss (margin=0.5) | Better geometry than contrastive |
| Similarity | Cosine + FAISS IndexFlatIP | Fast, exact, interpretable |
| Subset | 5–8 categories × 250 images | Balances training signal and compute |
| Embedding dim | 128 | Compact; retains discriminative power |

---

## Evaluation Metrics

- **Precision@K** — fraction of top-K results in the same category
- **Recall@K** — fraction of same-category items retrieved in top-K
- **Inference time** — per-query latency (ms)
- **Qualitative** — visual grid comparison: baseline vs. Siamese

---

## Model Comparison

Run the evaluation script to compare baseline (frozen ResNet50) vs. fine-tuned vs. Siamese:

```bash
python utils/evaluation.py \
    --data_dir data/subset \
    --baseline_model baseline \
    --siamese_model models/siamese_checkpoint.pt \
    --k 5
```
