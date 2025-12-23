"""Data module initialization."""

from .dataset import (
    DocumentDataset,
    DocumentDatasetWithRetrieval,
    collate_fn_classification,
    collate_fn_retrieval,
)
from .ocr import OCRProcessor, normalize_text, save_ocr_metadata, load_ocr_metadata

__all__ = [
    "DocumentDataset",
    "DocumentDatasetWithRetrieval",
    "collate_fn_classification",
    "collate_fn_retrieval",
    "OCRProcessor",
    "normalize_text",
    "save_ocr_metadata",
    "load_ocr_metadata",
]
