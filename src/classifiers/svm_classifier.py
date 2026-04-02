"""
SVM classifier with 5-fold cross-validation pipeline
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    classification_report, average_precision_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC

from src.utils.config import Config
from src.encoders.bag_of_words import BagOfWords
from src.encoders.vlad import VLAD


class SVMClassifier:
    """
    SVM klasifikator sa StandardScaler-om u Pipeline-u.
    """

    def __init__(
            self,
            C: float = Config.SVM_C,
            kernel: str = Config.SVM_KERNEL,
            gamma: str = Config.SVM_GAMMA,
            random_state: int = Config.RANDOM_STATE,
    ):
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(
                kernel=kernel,
                C=C,
                gamma=gamma,
                probability=True,
                random_state=random_state,
                class_weight='balanced',
            )),
        ])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVMClassifier":
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict_proba(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray, cv: int = Config.CV_FOLD) -> dict:
        """Evaluacija sa Stratified K-Fold CV. Vraća rečnik metrika."""
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=Config.RANDOM_STATE)
        fold_accs, fold_f1s = [], []
        all_y_test, all_y_pred, all_y_prob = [], [], []

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            clf = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', self.pipeline.named_steps['clf'].__class__(
                    **{k: v for k, v in self.pipeline.named_steps['clf'].get_params().items()}
                )),
            ])
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)

            fold_accs.append(accuracy_score(y_test, y_pred))
            fold_f1s.append(f1_score(y_test, y_pred, average='macro'))
            all_y_test.extend(y_test)
            all_y_pred.extend(y_pred)
            all_y_prob.append(y_prob)

        all_y_test = np.array(all_y_test)
        all_y_pred = np.array(all_y_pred)
        all_y_prob = np.vstack(all_y_prob)

        return {
            'mean_accuracy': float(np.mean(fold_accs)),
            'std_accuracy': float(np.std(fold_accs)),
            'mean_f1_macro': float(np.mean(fold_f1s)),
            'std_f1_macro': float(np.std(fold_f1s)),
            'fold_accuracies': fold_accs,
            'fold_f1s': fold_f1s,
            'classification_report': classification_report(
                all_y_test, all_y_pred, output_dict=True
            ),
            'confusion_matrix': confusion_matrix(all_y_test, all_y_pred).tolist(),
        }


def evaluate_classifier(
        X_raw: np.ndarray | list[np.ndarray],
        y: np.ndarray,
        class_names: list[str],
        experiment_name: str,
        save_path: str,
        encoding: str = 'direct',
        descriptor_type: str = 'global',
        n_folds: int = Config.CV_FOLD,
        random_state: int = Config.RANDOM_STATE,
        vocab_size: int = Config.BOW_VOCAB_SIZE,
        vlad_k: int = Config.VLAD_K,
) -> dict:
    """
    Kompletna evaluacija sa 5-Fold CV.

    Sve operacije koje uče iz podataka (BoW, VLAD, Scaler, SVM)
    se izvršavaju unutar CV petlje samo na trening skupu.

    Parametri:
        X_raw:           sirovi deskriptori pre kodovanja
                         global: [N, D]
                         local:  lista array-a [n_desc, D] po slici
        encoding:        'direct' → bez kodovanja
                         'bow'    → Bag of Words
                         'vlad'   → VLAD
        descriptor_type: 'global' → HOG, LBP, Gabor (jedan vektor po slici)
                         'local'  → SIFT, VGG lokalni (lista vektora po slici)
    """
    if descriptor_type == 'global':
        feat_dim = X_raw.shape[1]
        n_samples = X_raw.shape[0]
    else:
        feat_dim = X_raw[0].shape[1]
        n_samples = len(X_raw)

    print(f"\n{'=' * 60}")
    print(f"Eksperiment: {experiment_name}")
    print(f"Kodovanje: {encoding} | Tip: {descriptor_type}")
    print(f"Dimenzija sirovih deskriptora: {feat_dim} | Uzoraka: {n_samples}")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    indices = np.arange(n_samples)

    fold_accuracies, fold_f1s = [], []
    all_y_test, all_y_pred, all_y_prob = [], [], []
    fold_times = []
    encoded_dim = None

    for fold, (train_idx, test_idx) in enumerate(skf.split(indices, y)):
        if descriptor_type == 'global':
            X_train_raw, X_test_raw = X_raw[train_idx], X_raw[test_idx]
        else:
            X_train_raw = [X_raw[i] for i in train_idx]
            X_test_raw = [X_raw[i] for i in test_idx]

        y_train, y_test = y[train_idx], y[test_idx]

        if encoding == 'direct':
            X_train, X_test = X_train_raw, X_test_raw

        elif encoding == 'bow':
            bow = BagOfWords(vocab_size=vocab_size, random_state=random_state)
            if descriptor_type == 'global':
                bow.fit(X_train_raw)
                X_train = bow.transform_global(X_train_raw)
                X_test = bow.transform_global(X_test_raw)
            else:
                bow.fit(np.vstack(X_train_raw))
                X_train = bow.transform_local(X_train_raw)
                X_test = bow.transform_local(X_test_raw)

        elif encoding == 'vlad':
            if descriptor_type == 'global':
                print("VLAD nema smisla za globalne deskriptore – koristim direct.")
                X_train, X_test = X_train_raw, X_test_raw
            else:
                vlad = VLAD(k=vlad_k, random_state=random_state)
                vlad.fit(np.vstack(X_train_raw))
                X_train = vlad.transform(X_train_raw)
                X_test = vlad.transform(X_test_raw)
        else:
            raise ValueError(f"Nepoznato kodovanje: {encoding}")

        clf = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='rbf', C=10, gamma='scale',
                        probability=True, random_state=random_state,
                        class_weight='balanced')),
        ])

        t0 = time.time()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)
        fold_times.append(time.time() - t0)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        fold_accuracies.append(acc)
        fold_f1s.append(f1)
        all_y_test.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_prob.append(y_prob)

        print(f"  Fold {fold + 1}/{n_folds}: Acc={acc:.4f}, F1={f1:.4f}, "
              f"vreme={fold_times[-1]:.1f}s")

        if encoded_dim is None:
            encoded_dim = X_train.shape[1] if hasattr(X_train, 'shape') else len(X_train[0])

    all_y_test = np.array(all_y_test)
    all_y_pred = np.array(all_y_pred)
    all_y_prob = np.vstack(all_y_prob)

    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    mean_f1 = np.mean(fold_f1s)
    std_f1 = np.std(fold_f1s)

    print(f"\nRezultati ({n_folds}-Fold CV):")
    print(f"  Accuracy:   {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"  F1 (macro): {mean_f1:.4f} ± {std_f1:.4f}")

    n_classes = len(class_names)
    y_test_bin = label_binarize(all_y_test, classes=range(n_classes))
    prauc_per_class = {}
    for cls in range(n_classes):
        ap = average_precision_score(y_test_bin[:, cls], all_y_prob[:, cls])
        prauc_per_class[class_names[cls]] = float(ap)

    mean_prauc = float(np.mean(list(prauc_per_class.values())))
    print(f"  PR-AUC (macro): {mean_prauc:.4f}")

    report = classification_report(all_y_test, all_y_pred,
                                   target_names=class_names, output_dict=True)
    cm = confusion_matrix(all_y_test, all_y_pred)

    os.makedirs(save_path, exist_ok=True)

    # Confusion matrix plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(f"Confusion Matrix – {experiment_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(f"{save_path}/confusion_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()

    results = {
        "experiment": experiment_name,
        "encoding": encoding,
        "raw_feature_dim": feat_dim,
        "encoded_feature_dim": encoded_dim,
        "n_samples": n_samples,
        "mean_accuracy": float(mean_acc),
        "std_accuracy": float(std_acc),
        "mean_f1_macro": float(mean_f1),
        "std_f1_macro": float(std_f1),
        "mean_prauc": mean_prauc,
        "prauc_per_class": prauc_per_class,
        "classification_report": report,
        "fold_accuracies": fold_accuracies,
        "fold_f1s": fold_f1s,
        "mean_training_time_s": float(np.mean(fold_times)),
        "confusion_matrix": cm.tolist(),
    }

    with open(f"{save_path}/results.json", 'w') as f:
        json.dump(results, f, indent=2)

    return results
