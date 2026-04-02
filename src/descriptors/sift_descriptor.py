"""
SIFT (Scale-Invariant Feature Transform) descriptor extraction
"""
from __future__ import annotations

import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import color

from src.utils.config import Config


def extract_sift_descriptors(
        images: np.ndarray,
        max_keypoints: int = Config.SIFT_N_KEYPOINTS,
) -> tuple[list[np.ndarray], list[int]]:
    """
    Ekstrakcija SIFT deskriptora pomoću OpenCV.

    SIFT je lokalni deskriptor koji:
      - Detektuje ključne tačke (keypoints) invarijantno na skalu i rotaciju
      - Opisuje lokalni patch oko svake ključne tačke 128-dimenzionalnim vektorom
        (4x4 prostorna mreža x 8 orijentacionih binova)

    Napomena: SIFT vraća LISTU deskriptora po slici (varijabilan broj keypointa),
    pa nije direktno upotrebljiv bez kodovanja (BoW/VLAD).

    Parametri:
        max_keypoints: maksimalan broj keypointa po slici (0 = bez ograničenja)

    Vraća:
        all_descriptors: lista array-a [n_kp, 128] po slici
        keypoint_counts: broj keypointa po slici
    """
    sift = cv2.SIFT_create(nfeatures=max_keypoints)

    all_descriptors: list[np.ndarray] = []
    keypoint_counts: list[int] = []

    for img in images:
        gray_uint8 = (color.rgb2gray(img) * 255).astype(np.uint8)
        keypoints, descriptors = sift.detectAndCompute(gray_uint8, None)

        if descriptors is None or len(descriptors) == 0:
            descriptors = np.zeros((1, 128), dtype=np.float32)
            keypoints = []

        all_descriptors.append(descriptors.astype(np.float32))
        keypoint_counts.append(len(keypoints))

    return all_descriptors, keypoint_counts


def visualize_sift(
        images: np.ndarray,
        labels: np.ndarray,
        class_names: list[str],
        save_path: str,
        n_keypoints: int = Config.SIFT_N_KEYPOINTS,
        n_examples: int = 3,
) -> None:
    """
    Vizualizacija SIFT keypointa — originalna slika i slika s ucrtanim keypoitima.
    """
    sift = cv2.SIFT_create(nfeatures=n_keypoints)
    unique_cls = np.unique(labels)

    fig, axes = plt.subplots(len(unique_cls), n_examples * 2,
                             figsize=(4 * n_examples * 2, 3 * len(unique_cls)))

    for i, cls in enumerate(unique_cls):
        idx = np.where(labels == cls)[0][:n_examples]

        for j, id_ in enumerate(idx):
            gray_uint8 = (color.rgb2gray(images[id_]) * 255).astype(np.uint8)
            keypoints, _ = sift.detectAndCompute(gray_uint8, None)
            img_uint8 = (images[id_] * 255).astype(np.uint8)
            vis = cv2.drawKeypoints(img_uint8, keypoints, None,
                                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            axes[i, j * 2].imshow(images[id_])
            axes[i, j * 2].set_title(f"{class_names[cls]}" if j == 0 else "", fontsize=8)
            axes[i, j * 2].axis('off')
            axes[i, j * 2 + 1].imshow(vis)
            axes[i, j * 2 + 1].set_title("SIFT keypoints" if j == 0 else "", fontsize=8)
            axes[i, j * 2 + 1].axis('off')

    plt.suptitle('SIFT vizualizacija keypointa', fontsize=14)
    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f"{save_path}/sift_visualization.png", dpi=150, bbox_inches='tight')
    plt.close()
