# CropVision — Data-Efficient Crop Image Classification

## Overview

CropVision is a crop image classification project that investigates how pretrained vision foundation models can reduce the amount of labeled data required for effective image classification.

The pipeline uses a pretrained DINOv3 vision transformer to extract visual embeddings from crop images and trains lightweight machine learning classifiers on those embeddings. Instead of training a large vision model from scratch, it transfers pretrained visual representations to a crop-specific classification task.

The primary objective is to **reduce labeled-data requirements while maintaining strong classification performance**, using a computationally accessible pipeline that runs on a CPU.

## Key Features

- **Data-efficient classification:** Evaluates model performance using different fractions of the available labeled training data.
- **Pretrained DINOv3 embeddings:** Extracts normalized, 384-dimensional image representations.
- **Lightweight classifiers:** Compares MLP, RBF SVM, K-Nearest Neighbors, and Random Forest.
- **CPU-compatible workflow:** Supports embedding extraction, classifier training, and inference without requiring a GPU.
- **Stratified dataset splitting:** Maintains class representation across training and test subsets.
- **Rice class consolidation:** Maps Kharif rice and Rabi rice to a single rice class.
- **Embedding audit:** Uses cosine similarity and nearest-neighbor analysis to investigate potentially confusing images.
- **Reproducible experiments:** Uses fixed random seeds and a held-out test set for consistent comparisons.

## Methodology

### 1. Dataset preparation

Images are loaded from class-specific directories. Kharif rice and Rabi rice are consolidated into one rice category, resulting in six target classes.

The dataset is divided into training and test subsets using a stratified split with a fixed random seed.

### 2. Feature extraction using DINOv3

A pretrained DINOv3 vision transformer is used as a feature extractor.

Each image is converted into a 384-dimensional embedding, which is L2-normalized and saved for downstream experiments.

The vision backbone remains frozen; the current implementation does not fine-tune the entire transformer on the crop dataset.

### 3. Lightweight classifier training

Four machine learning classifiers are trained using the extracted embeddings:

- Multilayer Perceptron (MLP)
- Radial Basis Function Support Vector Machine (RBF SVM)
- K-Nearest Neighbors (KNN)
- Random Forest

The classifiers are evaluated using accuracy, balanced accuracy, and macro F1-score.

### 4. Training-data efficiency evaluation

The MLP classifier is evaluated using 25%, 50%, 75%, and 100% of the available training embeddings.

Each data fraction is evaluated across five random seeds while retaining the same held-out test set. This experiment measures how classification performance changes as the number of labeled training images decreases.

## Dataset

The current dataset contains 394 valid images across six classes.

| Crop class | Images |
|---|---:|
| Cotton | 55 |
| Gram | 46 |
| Mustard | 45 |
| Rice | 97 |
| Soybean | 74 |
| Wheat | 77 |
| **Total** | **394** |

The current stratified split contains:

- Training set: 334 images
- Test set: 60 images

The dataset is relatively small compared with datasets typically used to train vision models from scratch. The project therefore leverages pretrained visual representations to support classification with fewer task-specific labeled examples.

## Results

### Classifier comparison

The following results were obtained using the fixed 60-image test set.

| Classifier | Accuracy | Balanced Accuracy | Macro F1 |
|---|---:|---:|---:|
| MLP | **91.67%** | **90.69%** | **90.74%** |
| RBF SVM | 88.33% | 87.20% | 87.69% |
| Random Forest | 83.33% | 81.24% | 82.44% |
| KNN | 83.33% | 79.95% | 80.64% |

The MLP achieved the highest accuracy in this classifier comparison, correctly classifying 55 of the 60 test images in the reported run.

### Training-data efficiency

The following experiment evaluates the MLP with different fractions of the training set. Each configuration is run with five random seeds, and the same test set is used throughout.

| Training data | Training images | Mean accuracy | Accuracy std. dev. | Mean macro F1 |
|---|---:|---:|---:|---:|
| 25% | 83 | 73.00% | 4.47 pp | 72.47% |
| 50% | 167 | 84.67% | 3.80 pp | 84.40% |
| 75% | 250 | 87.00% | 2.17 pp | 86.52% |
| 100% | 334 | 90.67% | 2.24 pp | 89.87% |

*Standard deviations are calculated across five runs. `pp` denotes percentage points.*

The results show that classification performance improves as more labeled training examples become available.

Notably, using 50% of the training data achieved 84.67% mean accuracy, compared with 90.67% using the complete training set. Using 75% of the data achieved 87.00% mean accuracy.

These results provide initial evidence that pretrained DINOv3 embeddings can support useful crop classification with fewer labeled examples. However, the results also show that reducing the training set to 25% substantially decreases performance.

The experiment demonstrates data efficiency within the current dataset and evaluation setup; it does not establish equivalent performance on independently collected datasets or different agricultural regions.

## Project Structure

```text
cropvision/
├── 1_Training/
│   ├── __init__.py
│   ├── data.py
│   ├── embedding/
│   │   ├── __init__.py
│   │   ├── dinov3.py
│   │   └── extract.py
│   └── classifier/
│       ├── __init__.py
│       ├── train.py
│       └── models/
│           ├── __init__.py
│           ├── mlp.py
│           ├── rbf_svm.py
│           ├── knn.py
│           └── random_forest.py
├── 2_Inference/
│   └── predict.py
├── 3_Experiments/
│   ├── classifier_comparison/
│   ├── data_efficiency/
│   ├── embedding_audit/
│   └── independent_evaluation/
├── 4_Tests/
├── configs/
├── Dataset/
├── artifacts/
├── pyproject.toml
├── uv.lock
└── README.md
```

**Local-only directories:**

- `Dataset/` contains the local image dataset.
- `artifacts/` contains generated embeddings and trained model files.
- Experiment `results/` directories contain generated metrics, CSV files, and plots.

These directories are excluded from version control and are not required to be committed to the repository.

## Setup

### Prerequisites

- Python version supported by the project configuration
- [uv](https://docs.astral.sh/uv/)
- Internet connectivity for the initial pretrained-model download

Install the project dependencies:

```bash
uv sync
```

### Dataset organization

Place the labeled images in the following directory structure:

```text
Dataset/
└── images/
    ├── cotton/
    ├── gram/
    ├── kharif-rice/
    ├── mustard/
    ├── rabi-rice/
    ├── soybean/
    └── wheat/
```

The data loader maps both rice directories to the common `rice` label.

## Running the Pipeline

Run commands from the repository root.

### 1. Validate the dataset

```bash
uv run python 1_Training/data.py
```

### 2. Extract DINOv3 embeddings

```bash
uv run python 1_Training/embedding/extract.py
```

The training and test embeddings are saved to:

```text
artifacts/embeddings/crop_embeddings.npz
```

The split is performed before embedding extraction so that the saved training and test embeddings correspond to separate image subsets.

### 3. Compare classifiers

```bash
uv run python 3_Experiments/classifier_comparison/compare_classifiers.py
```

This trains and evaluates the configured classifiers and saves model artifacts and comparison results locally.

### 4. Run the data-efficiency experiment

```bash
uv run python 3_Experiments/data_efficiency/evaluate_data_efficiency.py
```

The script evaluates four training-data fractions across five random seeds.

Outputs are saved under:

```text
3_Experiments/data_efficiency/results/
```

The generated files include individual run metrics, summary statistics, and a plot comparing training-data size with classification performance.

### 5. Run inference on an image

```bash
uv run python 2_Inference/predict.py --image Dataset/images/soybean/img_0043.jpg
```

Replace the example path with the image you want to classify. The trained model and pretrained DINOv3 weights must be available locally.

### 6. Run the tests

```bash
uv run pytest 4_Tests -v
```

## Evaluation Metrics

- **Accuracy:** Fraction of test images classified correctly.
- **Balanced accuracy:** Average recall across classes, helping account for differences in class frequencies.
- **Macro F1-score:** Unweighted average of the F1-score across classes.
- **Standard deviation:** Measures variability in the reported metric across the five experimental runs.

The classifier comparison and data-efficiency evaluation use the same held-out test subset. Because this test set contains only 60 images, small changes in the number of correct predictions can noticeably affect the reported metrics.

## Limitations

- The dataset contains only 394 images, limiting the strength of conclusions about generalization.
- The held-out test set contains 60 images and may not represent the full variability of agricultural imagery.
- The data-efficiency results measure performance on the existing test set, not on independent farms, geographic regions, seasons, or imaging conditions.
- The experiment evaluates a pretrained DINOv3 feature extractor with lightweight classifiers; it does not establish superiority over every alternative feature extractor or end-to-end training approach.
- Independent evaluation requires a separate labeled dataset that was not used to develop the pipeline.

## Future Work

- Evaluate data efficiency on an independently collected dataset.
- Investigate performance across geographic regions and acquisition conditions.
- Analyze per-class precision, recall, and confusion matrices.
- Compare pretrained embeddings with alternative feature extraction approaches.
- Explore annotation-efficient learning strategies and deployment optimizations.

## Reproducibility and Data Usage

The repository contains the source code, configuration, and experiment scripts needed to reproduce the workflow. Dataset images, generated embeddings, and trained model artifacts are maintained locally rather than committed to Git.

Before redistributing the dataset or publishing derived assets, verify the applicable image licenses and attribution requirements.
