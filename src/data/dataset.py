"""PyTorch Dataset implementations."""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class DocumentDataset(Dataset):
    """Dataset for document images with text and labels."""

    def __init__(
        self,
        metadata_dir: str,
        image_dir: str,
        image_size: int = 224,
        max_text_length: int = 512,
        transform: Optional[transforms.Compose] = None,
    ):
        """Initialize dataset.

        Args:
            metadata_dir: Directory containing JSON metadata files
            image_dir: Directory containing document images
            image_size: Size to resize images to
            max_text_length: Maximum text length
            transform: Optional image transforms
        """
        self.metadata_dir = metadata_dir
        self.image_dir = image_dir
        self.max_text_length = max_text_length
        self.metadata_files = sorted(Path(metadata_dir).glob("*.json"))

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
        else:
            self.transform = transform

        self.label_to_idx = self._build_label_map()

    def _build_label_map(self) -> Dict[str, int]:
        """Build mapping from label strings to indices."""
        labels = set()
        for metadata_file in self.metadata_files:
            with open(metadata_file) as f:
                metadata = json.load(f)
                labels.add(metadata.get("label", "unknown"))

        return {label: idx for idx, label in enumerate(sorted(labels))}

    def __len__(self) -> int:
        return len(self.metadata_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, int]:
        """Get item by index.

        Returns:
            Tuple of (image_tensor, text, label_idx)
        """
        metadata_file = self.metadata_files[idx]
        with open(metadata_file) as f:
            metadata = json.load(f)

        # Load image
        image_path = Path(self.image_dir) / metadata["image_path"]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Get text and truncate
        text = metadata.get("text", "")
        text = text[: self.max_text_length]

        # Get label
        label = metadata.get("label", "unknown")
        label_idx = self.label_to_idx[label]

        return image, text, label_idx


class DocumentDatasetWithRetrieval(Dataset):
    """Dataset for joint image-text retrieval."""

    def __init__(
        self,
        metadata_dir: str,
        image_dir: str,
        image_size: int = 224,
        max_text_length: int = 512,
        transform: Optional[transforms.Compose] = None,
    ):
        """Initialize retrieval dataset."""
        self.base_dataset = DocumentDataset(
            metadata_dir=metadata_dir,
            image_dir=image_dir,
            image_size=image_size,
            max_text_length=max_text_length,
            transform=transform,
        )

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Dict:
        """Get item by index.

        Returns:
            Dict with image_tensor, text, and label_idx
        """
        image, text, label_idx = self.base_dataset[idx]
        return {
            "image": image,
            "text": text,
            "label": label_idx,
            "idx": idx,
        }


def collate_fn_classification(
    batch: List[Tuple[torch.Tensor, str, int]],
) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
    """Collate function for classification."""
    images, texts, labels = zip(*batch)
    images = torch.stack(images)
    labels = torch.tensor(labels)
    return images, list(texts), labels


def collate_fn_retrieval(batch: List[Dict]) -> Dict:
    """Collate function for retrieval."""
    images = torch.stack([item["image"] for item in batch])
    texts = [item["text"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch])
    indices = torch.tensor([item["idx"] for item in batch])

    return {
        "images": images,
        "texts": texts,
        "labels": labels,
        "indices": indices,
    }
