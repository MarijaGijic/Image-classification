"""
Bag of Words (BoW) visual vocabulary encoding
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from src.utils.config import Config


class BagOfWords:
    """
    Bag of Words kodovanje.

    fit() se uvek poziva samo na trening skupu (unutar CV petlje)
    da bi se izbegao data leakage.

    Podržava dva tipa deskriptora:
        - Globalni (HOG, LBP, Gabor): jedan vektor po slici → transform_global()
        - Lokalni (SIFT, VGG lokalni): lista vektora po slici → transform_local()
    """

    def __init__(self, vocab_size: int = Config.BOW_VOCAB_SIZE, random_state: int = 42):
        self.vocab_size = vocab_size
        self.kmeans = MiniBatchKMeans(
            n_clusters=vocab_size,
            random_state=random_state,
            n_init=10,
            max_iter=100,
        )
        self.fitted = False

    def fit(self, X: np.ndarray) -> "BagOfWords":
        """
        Trenira rečnik vizuelnih reči.

        X: [N, D] – svi deskriptori trening skupa
           Za globalne: X_train direktno
           Za lokalne:  np.vstack(sift_descs_train)

        Poziva se samo na trening skupu.
        """
        assert not self.fitted, "Rečnik je već treniran!"
        print(f"Treniranje BoW rečnika (k={self.vocab_size})...")
        t0 = time.time()
        self.kmeans.fit(X)
        print(f"  Gotovo za {time.time() - t0:.2f}s")
        self.fitted = True
        return self

    def transform_global(self, X_flat: np.ndarray) -> np.ndarray:
        """
        Koduje globalne deskriptore (HOG/LBP/Gabor).

        X_flat: [N, D] – jedan vektor po slici
        Vraća: [N, vocab_size] – one-hot BoW histogram po slici
        """
        assert self.fitted, "Pozovi fit() pre transform()!"
        bow_vectors = []
        for desc in X_flat:
            word = self.kmeans.predict(desc.reshape(1, -1))[0]
            hist = np.zeros(self.vocab_size, dtype=np.float32)
            hist[word] = 1.0
            bow_vectors.append(hist)
        return np.array(bow_vectors, dtype=np.float32)

    def transform_local(self, X_per_image: list[np.ndarray]) -> np.ndarray:
        """
        Koduje lokalne deskriptore (SIFT, VGG lokalni).

        X_per_image: lista array-a oblika [n_desc, D] po slici
        Vraća: [N, vocab_size] – normalizovani BoW histogram po slici
        """
        assert self.fitted, "Pozovi fit() pre transform()!"
        bow_vectors = []
        for descs in X_per_image:
            words = self.kmeans.predict(descs)
            hist, _ = np.histogram(
                words,
                bins=self.vocab_size,
                range=(0, self.vocab_size),
                density=True,
            )
            bow_vectors.append(hist.astype(np.float32))
        return np.array(bow_vectors, dtype=np.float32)
