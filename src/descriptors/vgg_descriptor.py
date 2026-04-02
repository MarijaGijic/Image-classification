"""
VGG16 middle-layer feature extraction using PyTorch
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torchvision.models as models
import torchvision.transforms as transforms

from src.utils.config import Config


class VGG16MiddleLayerExtractor:
    """
    Ekstrakcija feature mapa iz SREDNJIH slojeva VGG16.

    Umesto avgpool (skoro krajnji sloj), uzimamo feature mape
    iz konvolucionih slojeva koji sadrže prostorno bogate reprezentacije.

    Ove feature mape se mogu koristiti:
        - direktno (mean/max pooling po prostornoj dimenziji)
        - kao lokalni deskriptori za BoW/VLAD (svaka prostorna lokacija = jedan deskriptor)
    """

    def __init__(
            self,
            layers: list[str] = None,
            batch_size: int = Config.VGG_BATCH_SIZE,
            device: str | None = None,
    ):
        """
        layers: indeksi slojeva iz model.features
                '12' → Conv3_2  (56×56×256) – srednji sloj
                '19' → Conv4_2  (28×28×512) – srednji-duboki sloj
                '24' → Conv5_1  (14×14×512) – dublji sloj
        """
        if layers is None:
            layers = ['12', '19', '24']

        self.layer_names = layers
        self.batch_size = batch_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Koristim: {self.device}")

        self.model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.model.eval()
        self.model.to(self.device)

        self.layer_outputs: dict[str, np.ndarray] = {}

        for layer_idx in layers:
            layer = self.model.features[int(layer_idx)]
            layer.register_forward_hook(self._make_hook(layer_idx))

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def _make_hook(self, layer_idx: str):
        def hook(module, input, output):
            self.layer_outputs[layer_idx] = output.detach().cpu().numpy()
        return hook

    def extract_global(self, images: np.ndarray) -> np.ndarray:
        """
        Globalna ekstrakcija – mean + max pooling po prostornoj dimenziji.

        Svaka feature mapa (C, H, W) → mean po H,W → jedan broj po kanalu
        Rezultat: vektor fiksne dužine po slici, spreman za direktnu klasifikaciju.

        Primer za sloj '12' (256, 56, 56):
            mean pooling → (256,)
            max pooling  → (256,)
            konkatenirano → (512,) po sloju
        """
        all_features = []

        for i in tqdm(range(0, len(images), self.batch_size), desc="VGG16 srednji slojevi"):
            batch = images[i:i + self.batch_size]
            tensors = [
                self.transform(Image.fromarray((img * 255).astype(np.uint8)))
                for img in batch
            ]
            batch_tensor = torch.stack(tensors).to(self.device)

            with torch.no_grad():
                _ = self.model(batch_tensor)

            batch_features = []
            for layer_idx in self.layer_names:
                feat_map = self.layer_outputs[layer_idx]  # (batch, C, H, W)
                mean_pool = feat_map.mean(axis=(2, 3))    # (batch, C)
                max_pool = feat_map.max(axis=(2, 3))      # (batch, C)
                pooled = np.concatenate([mean_pool, max_pool], axis=1)
                batch_features.append(pooled)

            combined = np.concatenate(batch_features, axis=1)
            all_features.append(combined)

        return np.vstack(all_features)

    def extract_local(self, images: np.ndarray) -> list[np.ndarray]:
        """
        Lokalna ekstrakcija – svaka prostorna lokacija = jedan deskriptor.

        Feature mapa (C, H, W) → H×W lokalnih deskriptora dimenzije C.
        Ovo je analogno SIFT keypoint deskriptorima → može kroz BoW/VLAD!

        Primer za sloj '24' (512, 14, 14):
            → 14×14 = 196 lokalnih deskriptora dimenzije 512
        """
        all_local_descriptors: list[np.ndarray] = []

        for i in tqdm(range(0, len(images), self.batch_size), desc="VGG16 lokalni deskriptori"):
            batch = images[i:i + self.batch_size]
            tensors = [
                self.transform(Image.fromarray((img * 255).astype(np.uint8)))
                for img in batch
            ]
            batch_tensor = torch.stack(tensors).to(self.device)

            with torch.no_grad():
                _ = self.model(batch_tensor)

            last_layer = self.layer_names[-1]
            feat_map = self.layer_outputs[last_layer]  # (batch, C, H, W)
            batch_size_actual = feat_map.shape[0]
            C = feat_map.shape[1]

            for b in range(batch_size_actual):
                local_descs = feat_map[b].reshape(C, -1).T  # (H*W, C)
                all_local_descriptors.append(local_descs.astype(np.float32))

        return all_local_descriptors
