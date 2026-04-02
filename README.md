# Image Classification

Experimental benchmarking of classical computer vision descriptors and deep CNN features as inputs to an SVM classifier, evaluated on 5 selected classes from the Caltech-256 Object Categories dataset.

## Dataset

[Caltech-256](https://www.kaggle.com/datasets/jessicali9530/caltech256) — 5 selected categories:
- backpack
- dog
- dolphin
- faces-easy
- soccer-ball

Set the dataset path via environment variable or edit `src/utils/config.py`:
```bash
export DATASET_PATH=/path/to/256_ObjectCategories
```

## Feature Descriptors

| Descriptor | Type | Module |
|-----------|------|--------|
| HOG | Global | `src/descriptors/hog_descriptor.py` |
| LBP | Global | `src/descriptors/lbp_descriptor.py` |
| Gabor | Global | `src/descriptors/gabor_descriptor.py` |
| SIFT | Local | `src/descriptors/sift_descriptor.py` |
| VGG16 (middle layers) | Global / Local | `src/descriptors/vgg_descriptor.py` |

## Encoding Methods

| Encoder | Module |
|---------|--------|
| Bag of Words (BoW) | `src/encoders/bag_of_words.py` |
| VLAD | `src/encoders/vlad.py` |

## Classifier

SVM with RBF kernel, 5-fold stratified cross-validation — `src/classifiers/svm_classifier.py`.

## Project Structure

```
image-classification/
├── classification.ipynb     # Full pipeline notebook
├── src/
│   ├── descriptors/         # Feature extraction modules
│   ├── encoders/            # BoW and VLAD encoding
│   ├── classifiers/         # SVM + CV evaluation
│   └── utils/               # Config, data loader, visualization
```

## Running

Open and run `classification.ipynb` end-to-end. Results (plots, JSON metrics) are saved to `results/`.
