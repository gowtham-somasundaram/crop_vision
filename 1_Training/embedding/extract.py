
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIR = PROJECT_ROOT / "1_Training"

if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from data import load_image_paths, split_dataset
from embedding.dinov3 import DINOv3Embedder


DATASET_DIR = PROJECT_ROOT / "Dataset" / "images"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "embeddings"


def main():
    image_paths, labels = load_image_paths(DATASET_DIR)

    X_train_paths, X_test_paths, y_train, y_test = split_dataset(
        image_paths,
        labels,
        test_size=0.15,
        random_state=42,
    )

    print(f"Training images: {len(X_train_paths)}")
    print(f"Test images: {len(X_test_paths)}")

    embedder = DINOv3Embedder()

    X_train = embedder.embed_images(X_train_paths).numpy()
    X_test = embedder.embed_images(X_test_paths).numpy()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "crop_embeddings.npz"

    np.savez_compressed(
        output_path,
        X_train=X_train,
        y_train=np.asarray(y_train),
        paths_train=np.asarray(
            [p.relative_to(PROJECT_ROOT).as_posix() for p in X_train_paths]
        ),
        X_test=X_test,
        y_test=np.asarray(y_test),
        paths_test=np.asarray(
            [p.relative_to(PROJECT_ROOT).as_posix() for p in X_test_paths]
        ),
    )

    print(f"Training embeddings: {X_train.shape}")
    print(f"Test embeddings: {X_test.shape}")
    print(f"Saved to: {output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
