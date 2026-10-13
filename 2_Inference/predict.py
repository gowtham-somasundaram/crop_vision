
import argparse
from pathlib import Path

import joblib
import torch

from PIL import Image

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = PROJECT_ROOT / "1_Training"

if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from embedding.dinov3 import DINOv3Embedder


MODEL_PATH = PROJECT_ROOT / "artifacts" / "models" / "mlp.joblib"


def predict_image(image_path: Path, model_path: Path = MODEL_PATH):
    """Predict the crop class for a single image."""
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model not found: {model_path}. "
            "Run the classifier comparison or training script first."
        )

    with Image.open(image_path) as image:
        image.verify()

    embedder = DINOv3Embedder()

    embedding = embedder.embed_images([image_path]).numpy()

    model = joblib.load(model_path)
    prediction = model.predict(embedding)[0]

    result = {"image": str(image_path), "predicted_class": str(prediction)}

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(embedding)[0]
        result["class_scores"] = {
            str(label): round(float(score), 4)
            for label, score in zip(model.classes_, probabilities)
        }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Predict a crop class from an image using DINOv3 and MLP."
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the input image, relative to the project root or absolute.",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.is_absolute():
        image_path = PROJECT_ROOT / image_path

    result = predict_image(image_path)

    print(f"Image: {result['image']}")
    print(f"Predicted crop: {result['predicted_class']}")

    if "class_scores" in result:
        print("Model class scores:")
        for label, score in sorted(
            result["class_scores"].items(),
            key=lambda item: item[1],
            reverse=True,
        ):
            print(f"  {label}: {score:.4f}")


if __name__ == "__main__":
    main()
