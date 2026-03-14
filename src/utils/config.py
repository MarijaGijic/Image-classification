"""
Global configuration
"""
import os
import numpy as np

class Config:
    DATASET_PATH: str = os.environ.get("DATASET_PATH", 
                                       "/root/.cache/kagglehub/datasets/jessicali9530/caltech256/versions/2/256_ObjectCategories")
    SELECTED_CATEGORIES: list[str] = [
        "037.backpack",
        "057.dog",
        "065.dolphin-101",
        "066.faces-easy-101",
        "196.soccer-ball",
    ]
    IMG_SIZE: tuple[int, int] = (128, 128)

    RESULT_BASE: str = "results"
    # HoG
    HOG_PIXELS_PER_CELL: tuple[int, int] = (8, 8)
    HOG_CELLS_PER_BLOCK: tuple[int, int] = (2, 2)
    HOG_ORIENTATION: int = 9

    # LBP
    LBP_RADIUS: float = 1.0
    LBP_N_POINTS: int = 8

    # Gabor
    GABOR_FREQS: list[float] = [0.1, 0.3, 0.5]
    GABOR_THETAS: list[float] = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    # SIFT
    SIFT_N_KEYPOINTS: int = 100

    # VGG
    VGG_BATCH_SIZE: int = 32

    # Encoding
    BOW_VOCAB_SIZE: int = 256
    VLAD_K: int = 64

    # Classification
    CV_FOLD: int = 5
    RANDOM_STATE: int = 42
    SVM_C: float = 10.0
    SVM_KERNEL: str = "rbf"
    SVM_GAMMA: str = "scale"