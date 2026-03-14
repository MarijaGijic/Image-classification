"""
Dataset loading and preprocessing
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from src.utils.config import Config

def load_dataset(
        dataset_path: str,
        categories: list[str],
        img_size: tuple[int, int] = Config.IMG_SIZE,
        max_per_class: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    
    images: list[np.ndarray] = []
    labels: list[str] = []
    category_stats: dict[str, int] = {}

    for cat_folder in categories:
        cat_path = Path(dataset_path) / cat_folder
        if not cat_path.exists():
            print(f"Warning: Category not found")
            continue

        cat_name = cat_folder.split(".", 1)[1] if "." in cat_folder else cat_folder
        files = [
            f for f in os.listdir(cat_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if max_per_class:
            files = files[:max_per_class]

        count = 0

        for fname in tqdm(files, desc=f"Loading {cat_name}"):
            fpath = cat_path / fname
            try:
                img = Image.open(fpath).convert("RGB")
                arr = (np.array(img.resize(img_size)) / 255.0).astype(np.float32)
                images.append(arr)
                labels.append(cat_name)
                count += 1
            except Exception as exc:
                print(f" Skipping {fname}: {exc}")

        category_stats[fname] = count
        print(f" {cat_name}: {count} images loaded")
    
    return np.array(images), np.array(labels), category_stats

def encode_labels(labels: np.ndarray) -> tuple[np.ndarray, LabelEncoder]:
    # convert string labels to integer indices
    le = LabelEncoder()
    y = le.fit_transform(labels)
    return y, le

def save_dataset_stats(stats: dict, save_path: str) -> None:
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "dataset_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)