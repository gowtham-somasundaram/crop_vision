
from pathlib import Path

import timm
import torch
from PIL import Image
from torchvision import transforms


class DINOv3Embedder:
    """Extract L2-normalized image embeddings using pretrained DINOv3."""

    def __init__(
        self,
        model_name: str = "vit_small_patch16_dinov3",
        device: str = "cpu",
    ):
        self.device = torch.device(device)

        self.model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
        )
        self.model.eval().to(self.device)

        config = timm.data.resolve_model_data_config(self.model)

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (config["input_size"][1], config["input_size"][2])
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=config["mean"],
                    std=config["std"],
                ),
            ]
        )

        self.embedding_dim = self.model.num_features

    @torch.inference_mode()
    def embed_images(
        self,
        image_paths: list[str | Path],
        batch_size: int = 16,
    ) -> torch.Tensor:
        """Return one normalized embedding per image."""
        if not image_paths:
            raise ValueError("image_paths cannot be empty.")

        if batch_size < 1:
            raise ValueError("batch_size must be positive.")

        all_embeddings = []

        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start : start + batch_size]
            tensors = []

            for path in batch_paths:
                with Image.open(path) as image:
                    tensors.append(
                        self.transform(image.convert("RGB"))
                    )

            batch = torch.stack(tensors).to(self.device)
            embeddings = self.model(batch)
            embeddings = torch.nn.functional.normalize(
                embeddings, p=2, dim=1
            )

            all_embeddings.append(embeddings.cpu())

            print(
                f"Processed {min(start + len(batch_paths), len(image_paths))}"
                f"/{len(image_paths)} images",
                end="\r",
            )

        print()
        return torch.cat(all_embeddings, dim=0)
