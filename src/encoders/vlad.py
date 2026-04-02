"""
VLAD – Vector of Locally Aggregated Descriptors
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from src.utils.config import Config


class VLAD:
    """
    VLAD – Vector of Locally Aggregated Descriptors.

    NAPOMENA: VLAD ima smisla SAMO za lokalne deskriptore
              (SIFT, VGG lokalni slojevi) gde svaka slika
              ima više deskriptora. Za globalne deskriptore
              (HOG, LBP, Gabor) koristiti direktno ili BoW.

    VAŽNO: fit() pozivati SAMO na trening skupu!
    """

    def __init__(self, k: int = Config.VLAD_K, random_state: int = 42):
        self.k = k
        self.kmeans = MiniBatchKMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=10,
            max_iter=100,
        )
        self.fitted = False
        self.centers: np.ndarray | None = None
        self.D: int | None = None

    def fit(self, X: np.ndarray) -> "VLAD":
        """
        Trenira VLAD rečnik centara.

        X: [N, D] – svi lokalni deskriptori trening skupa
           Primer: np.vstack(sift_descs_train)
        """
        assert not self.fitted, "Rečnik je već treniran!"
        print(f"Treniranje VLAD rečnika (k={self.k})...")
        t0 = time.time()
        self.kmeans.fit(X)
        self.centers = self.kmeans.cluster_centers_
        self.D = X.shape[1]
        print(f"  Gotovo za {time.time() - t0:.2f}s")
        self.fitted = True
        return self

    def transform(self, X_per_image: list[np.ndarray]) -> np.ndarray:
        """
        Koduje lokalne deskriptore u VLAD vektore.

        X_per_image: lista array-a [n_desc, D] po slici
        Vraća: [N, k*D] – L2-normalizovani VLAD vektor po slici
        """
        assert self.fitted, "Pozovi fit() pre transform()!"
        vlad_vectors = []

        for descs in X_per_image:
            if descs.ndim == 1:
                descs = descs.reshape(1, -1)

            assignments = self.kmeans.predict(descs)

            vlad = np.zeros((self.k, self.D), dtype=np.float64)
            for desc, word in zip(descs, assignments):
                vlad[word] += desc - self.centers[word]

            # Intra-normalizacija po centru
            for i in range(self.k):
                norm = np.linalg.norm(vlad[i])
                if norm > 1e-8:
                    vlad[i] /= norm

            # Globalna L2 normalizacija
            vlad_flat = vlad.flatten()
            norm = np.linalg.norm(vlad_flat)
            if norm > 1e-8:
                vlad_flat /= norm

            vlad_vectors.append(vlad_flat.astype(np.float32))

        return np.array(vlad_vectors, dtype=np.float32)
