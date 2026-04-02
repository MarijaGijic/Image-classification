"""
Gabor filter feature extraction
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import color
from skimage.filters import gabor_kernel


def build_gabor_filterbank(
        frequencies: list[float] = None,
        thetas: list[float] = None,
) -> tuple[list[np.ndarray], list[dict]]:
    """
    Kreira banku Gabor filtara za različite frekvencije i orijentacije.
    Gabor filtri su posebno efikasni za analizu teksture i detekciju ivica.
    """
    from src.utils.config import Config

    if frequencies is None:
        frequencies = Config.GABOR_FREQS
    if thetas is None:
        thetas = Config.GABOR_THETAS

    kernels = []
    params = []
    for freq in frequencies:
        for theta in thetas:
            kernel = np.real(gabor_kernel(freq, theta=theta, sigma_x=1, sigma_y=1))
            kernels.append(kernel)
            params.append({'frequency': freq, 'theta': theta})
    return kernels, params


def extract_gabor_features(
        images: np.ndarray,
        frequencies: list[float] = None,
        thetas: list[float] = None,
) -> np.ndarray:
    """
    Primenjuje Gabor filterbanku na svaku sliku i kreira vektor obeležja.
    Za svaki filter: mean i std odziva daju statistike teksture.
    """
    kernels, _ = build_gabor_filterbank(frequencies, thetas)
    features = []

    for img in images:
        gray = color.rgb2gray(img)
        img_features = []

        for kernel in kernels:
            filtered = ndimage.convolve(gray, kernel, mode='wrap')
            img_features.append(np.mean(filtered))
            img_features.append(np.std(filtered))

        features.append(np.array(img_features, dtype=np.float32))

    return np.array(features, dtype=np.float32)


def visualize_gabor(
        images: np.ndarray,
        labels: np.ndarray,
        class_names: list[str],
        save_path: str,
        n_examples: int = 2,
) -> None:
    """
    Vizualizacija Gabor odziva — kompjutira odzive interno za n_examples slika.
    """
    import os

    kernels, _ = build_gabor_filterbank()
    n_filters_show = min(6, len(kernels))

    unique_cls = np.unique(labels)
    example_ids = [np.where(labels == cls)[0][0] for cls in unique_cls[:n_examples]]

    fig, axes = plt.subplots(n_examples + 1, n_filters_show + 1,
                             figsize=(3 * (n_filters_show + 1), 3 * (n_examples + 1)))

    axes[0, 0].text(0.5, 0.5, 'Original', ha='center', va='center', fontsize=10)
    axes[0, 0].axis('off')
    for f in range(n_filters_show):
        axes[0, f + 1].text(0.5, 0.5, f'Filter {f + 1}', ha='center', va='center', fontsize=9)
        axes[0, f + 1].axis('off')

    for i, id_ in enumerate(example_ids):
        gray = color.rgb2gray(images[id_])
        axes[i + 1, 0].imshow(images[id_])
        axes[i + 1, 0].axis('off')

        for f, kernel in enumerate(kernels[:n_filters_show]):
            filtered = ndimage.convolve(gray, kernel, mode='wrap')
            axes[i + 1, f + 1].imshow(filtered, cmap='gray')
            axes[i + 1, f + 1].axis('off')

    plt.suptitle('Gabor filter odzivi', fontsize=14)
    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f"{save_path}/gabor_visualization.png", dpi=150, bbox_inches='tight')
    plt.close()
