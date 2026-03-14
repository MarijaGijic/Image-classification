"""
Visualisation utilities
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_dataset_distribution(
        category_stats: dict[str, int],
        save_path: str,
) -> None:
    cats = list(category_stats.keys())
    counts = list(category_stats.values())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(cats, counts, color="steelblue", edgecolor="black")
    axes[0].set_title("Images per category", fontsize=14)
    axes[0].set_xlabel("Category")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=25)
    for i, v in enumerate(counts):
        axes[0].text(i, v+1, str(v), ha="center", fontsize=11)
    
    axes[1].pie(counts, labels=cats, autopct="%1.1f%%", startangle=140)
    axes[1].set_title("Class distribution", fontsize=14)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

def plot_sample_images(
        images: np.ndarray,
        labels: np.ndarray,
        class_names: np.ndarray,
        save_path: str,
        n_per_class: int = 5,
) -> None:
    
    n_classes = len(class_names)
    _, axes = plt.subplots(n_classes, n_per_class, figsize=(n_per_class*3, n_classes*3))

    for i, label in enumerate(class_names):
        cls_idx = np.where(labels == i)[0][:n_per_class]
        for j, idx in enumerate(cls_idx):
            ax = axes[i, j] if n_classes > 1 else axes[j]
            ax.show(images[idx])
            ax.set_title(label if j == 0 else "", fontsize=9)
            ax.axis("off")

    plt.suptitle("Sample images per class", fontsize=15)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

