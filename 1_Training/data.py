
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "Dataset" / "images"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CLASS_MAPPING = {
    "kharif-rice": "rice",
    "rabi-rice": "rice",
}


def load_image_paths(data_dir: str | Path = DATASET_DIR):
    """Load valid image paths and normalized crop labels."""
    data_dir = Path(data_dir)

    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir

    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {data_dir}"
        )

    image_paths = []
    labels = []

    class_dirs = sorted(
        path for path in data_dir.iterdir() if path.is_dir()
    )

    if not class_dirs:
        raise ValueError(f"No class folders found in {data_dir}")

    for class_dir in class_dirs:
        label = CLASS_MAPPING.get(class_dir.name, class_dir.name)

        for image_path in sorted(class_dir.rglob("*")):
            if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            try:
                with Image.open(image_path) as image:
                    image.verify()
            except (OSError, UnidentifiedImageError):
                print(f"Skipping invalid image: {image_path}")
                continue

            image_paths.append(image_path)
            labels.append(label)

    if not image_paths:
        raise ValueError(f"No valid images found in {data_dir}")

    if len(set(labels)) < 2:
        raise ValueError("At least two classes are required.")

    return image_paths, labels


def split_dataset(
    image_paths,
    labels,
    test_size: float = 0.15,
    random_state: int = 42,
):
    """Return a reproducible, stratified train/test split."""
    if len(image_paths) != len(labels):
        raise ValueError(
            "Image paths and labels must have equal lengths."
        )

    return train_test_split(
        image_paths,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )


if __name__ == "__main__":
    paths, labels = load_image_paths()

    print(f"Total valid images: {len(paths)}")
    print("Class distribution:")

    for label in sorted(set(labels)):
        print(f"  {label}: {labels.count(label)}")

    X_train, X_test, y_train, y_test = split_dataset(paths, labels)

    print(f"\nTraining images: {len(X_train)}")
    print(f"Test images: {len(X_test)}")
