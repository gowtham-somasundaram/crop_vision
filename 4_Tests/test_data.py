
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = PROJECT_ROOT / "1_Training"

if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from data import load_image_paths, split_dataset, CLASS_MAPPING


def test_rice_classes_are_mapped_correctly():
    assert CLASS_MAPPING["kharif-rice"] == "rice"
    assert CLASS_MAPPING["rabi-rice"] == "rice"


def test_dataset_loading_returns_matching_paths_and_labels():
    dataset_dir = PROJECT_ROOT / "Dataset" / "images"

    if not dataset_dir.is_dir():
        pytest.skip("Local dataset is not available.")

    paths, labels = load_image_paths(dataset_dir)

    assert len(paths) > 0
    assert len(paths) == len(labels)
    assert set(labels) == {
        "cotton",
        "gram",
        "mustard",
        "rice",
        "soybean",
        "wheat",
    }


def test_split_is_reproducible():
    paths = [f"image_{i}.jpg" for i in range(60)]
    labels = ["cotton"] * 10 + ["gram"] * 10 + ["rice"] * 10
    labels += ["soybean"] * 10 + ["wheat"] * 10 + ["mustard"] * 10

    first = split_dataset(paths, labels, test_size=0.2, random_state=42)
    second = split_dataset(paths, labels, test_size=0.2, random_state=42)

    assert first == second


def test_split_preserves_all_samples():
    paths = [f"image_{i}.jpg" for i in range(60)]
    labels = ["cotton"] * 10 + ["gram"] * 10 + ["rice"] * 10
    labels += ["soybean"] * 10 + ["wheat"] * 10 + ["mustard"] * 10

    X_train, X_test, y_train, y_test = split_dataset(
        paths, labels, test_size=0.2, random_state=42
    )

    assert len(X_train) + len(X_test) == 60
    assert len(y_train) + len(y_test) == 60
    assert set(X_train).isdisjoint(set(X_test))
