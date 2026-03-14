"""
Histogram of Oriented Gradients feature extraction
"""

from __future__ import annotations

import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.utils.config import Config

def extract_hog_features(
        images: np.ndarray,
        pixels_per_cell: tuple[int, int] = Config.HOG_PIXELS_PER_CELL,
        cells_per_block: tuple[int, int] = Config.HOG_CELLS_PER_BLOCK,
        orientations: int = Config.HOG_ORIENTATION,
) -> np.ndarray:
    """
    Extract HoG feature vectors for a batch of images
    """
    h, w = images.shape[1], images.shape[2]

    hog = cv2.HOGDescriptor(
        _winSize = (w, h),
        _blockSize=(cells_per_block[1] * pixels_per_cell[1],
                    cells_per_block[0] * pixels_per_cell[0]),
        _blockStride=(pixels_per_cell[1], pixels_per_cell[0]),
        _cellSize=(pixels_per_cell[1], pixels_per_cell[0]),
        _nbins=orientations,
    )

    features = []
    t0 = time.time()

    for img in images:
        gray = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        feat = hog.compute(gray).flatten()
        features.append(feat)
    
    elapsed = time.time() - t0
    X = np.array(features, dtype=np.float32)
    print(f"HOG features: {X.shape}  |  {elapsed:.2f}s total  |  {elapsed/len(images)*1000:.2f}ms/image")
    return X

def visualize_hog(
        images: np.ndarray,
        labels: np.ndarray,
        class_names: np.ndarray,
        save_path: str,
        n_per_class: int = 2,
) -> None:
    
    h, w = images.shape[1], images.shape[2]
    ppc = Config.HOG_PIXELS_PER_CELL
    cpb = Config.HOG_CELLS_PER_BLOCK

    hog = cv2.HOGDescriptor(
        _winsize=(w, h),
        _blockSize = (cpb[1]*ppc[1], cpb[0]*ppc[0]),
        _blockStride = (ppc[1], ppc[0]),
        _cellSize = (ppc[1], ppc[0]),
        _nbins = Config.HOG_ORIENTATION,
    )

    n_classes = len(class_names)
    fig, axes = plt.subplots(n_classes * n_per_class, 2,
    figsize = (6, 3 * n_classes * n_per_class),
    )

    row = 0
    for cls_idx, cls_name in enumerate(class_names):
        idxs = np.where(labels == cls_idx)[0][:n_per_class]
        for img_idx in idxs:
            img = images[img_idx]
            img_uint8 = (img*255).astype(np.uint8)
            gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)

            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
            mag, _ = cv2.cartToPolar(gx, gy)
            mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            axes[row, 0].imshow(img)
            axes[row, 0].set_title(f"{cls_name} - original", fontsize=8)
            axes[row, 0].axis("off")
            axes[row, 1].imshow(mag, cmap="gray")
            axes[row, 1].set_title("HOG gradients", fontsize=8)
            axes[row, 1].axis("off")
            row+=1
    plt.suptitle("HOG Visualisation", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"HOG visualisation saved to {save_path}")