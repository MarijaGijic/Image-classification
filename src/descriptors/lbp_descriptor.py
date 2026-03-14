"""
LBP (Local Binary Patterns) feature extraction
"""
from __future__ import annotations

import time

import numpy as np
from skimage.color import rgb2gray
from skimage.feature import local_binary_pattern

from src.utils.config import Config

def extract_lbp_features(
        images: np.ndarray,
        radius: float = Config.LBP_RADIUS,
        n_points: int = Config.LBP_N_POINTS,
) -> np.ndarray:
    
    features = []
    n_bins = 59
    t0 = time.time()

    for img in images:
        gray = rgb2gray(img)
        lbp = local_binary_pattern(img, P=n_points, R=radius, method="uniform")
        hist, _ = np.histogram(lbp.ravel(), bins=59, range=(0, 59), density=True)
        features.append(hist.astype(np.float32))
    
    elapsed = time.time() - t0
    X = np.array(features, dtype=np.float32)
    print(f"LBP features: {X.shape}  |  {elapsed:.2f}s total  |  {elapsed/len(images)*1000:.2f}ms/image")
    return X