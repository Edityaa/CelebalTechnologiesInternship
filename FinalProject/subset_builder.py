"""
data/subset_builder.py
----------------------
Builds a balanced training subset from the Fashion Product Images dataset.

Usage:
    python data/subset_builder.py \
        --dataset_dir data/fashion-dataset \
        --output_dir  data/subset \
        --categories  "Shirts,Shoes,Dresses,Watches,Bags,Sunglasses,Jeans" \
        --samples_per_category 250
"""

import argparse
import os
import shutil
import random
from pathlib import Path

import pandas as pd
from tqdm import tqdm


DEFAULT_CATEGORIES = [
    "Shirts", "Shoes", "Dresses", "Watches", "Bags", "Sunglasses", "Jeans"
]


def build_subset(
    dataset_dir: str,
    output_dir: str,
    categories: list[str],
    samples_per_category: int,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Copies a stratified subset of images into output_dir/<category>/ folders
    and returns a metadata DataFrame with columns [id, category, split, image_path].

    Args:
        dataset_dir:            Root of the Kaggle fashion-dataset download.
        output_dir:             Where to write the subset (created if missing).
        categories:             List of article-type category names to include.
        samples_per_category:   Max images per category.
        seed:                   Random seed for reproducibility.

    Returns:
        DataFrame with one row per image.
    """
    random.seed(seed)

    styles_csv = Path(dataset_dir) / "styles.csv"
    images_dir = Path(dataset_dir) / "images"

    if not styles_csv.exists():
        raise FileNotFoundError(
            f"styles.csv not found at {styles_csv}. "
            "Please download the dataset from:\n"
            "  https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset"
        )

    print(f"Loading styles metadata from {styles_csv} ...")
    df = pd.read_csv(styles_csv, on_bad_lines="skip")
    df = df.dropna(subset=["id", "articleType"])
    df["id"] = df["id"].astype(int)

    # ── filter to requested categories ──────────────────────────────────────
    df = df[df["articleType"].isin(categories)].copy()
    missing = set(categories) - set(df["articleType"].unique())
    if missing:
        print(f"  ⚠  Categories not found in dataset: {missing}")

    # ── sample per category ──────────────────────────────────────────────────
    sampled_rows = []
    for cat, group in df.groupby("articleType"):
        ids = group["id"].tolist()
        chosen = random.sample(ids, min(samples_per_category, len(ids)))
        sampled_rows.append(group[group["id"].isin(chosen)])
        print(f"  {cat}: {len(chosen)} images selected")

    subset_df = pd.concat(sampled_rows, ignore_index=True)

    # ── copy images and build manifest ──────────────────────────────────────
    output_path = Path(output_dir)
    records = []

    for _, row in tqdm(subset_df.iterrows(), total=len(subset_df), desc="Copying images"):
        img_id = row["id"]
        category = row["articleType"]
        src = images_dir / f"{img_id}.jpg"

        if not src.exists():
            continue  # image file missing in download; skip gracefully

        dest_dir = output_path / category
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{img_id}.jpg"
        shutil.copy2(src, dest)

        records.append({"id": img_id, "category": category, "image_path": str(dest)})

    manifest = pd.DataFrame(records)

    # ── train/val split (80/20 per category) ────────────────────────────────
    manifest["split"] = "train"
    for _, group in manifest.groupby("category"):
        val_idx = group.sample(frac=0.2, random_state=seed).index
        manifest.loc[val_idx, "split"] = "val"

    manifest.to_csv(output_path / "manifest.csv", index=False)

    total = len(manifest)
    train_n = (manifest["split"] == "train").sum()
    val_n = (manifest["split"] == "val").sum()
    print(
        f"\n✅ Subset ready at '{output_dir}'\n"
        f"   Total: {total}  |  Train: {train_n}  |  Val: {val_n}\n"
        f"   Manifest saved to {output_path / 'manifest.csv'}"
    )
    return manifest


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build a subset of the Fashion dataset")
    parser.add_argument("--dataset_dir", required=True, help="Root of the Kaggle download")
    parser.add_argument("--output_dir", default="data/subset", help="Where to write the subset")
    parser.add_argument(
        "--categories",
        default=",".join(DEFAULT_CATEGORIES),
        help="Comma-separated list of article types to include",
    )
    parser.add_argument("--samples_per_category", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    categories = [c.strip() for c in args.categories.split(",")]

    build_subset(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        categories=categories,
        samples_per_category=args.samples_per_category,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
