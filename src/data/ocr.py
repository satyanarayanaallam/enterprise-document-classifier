"""Data processing and OCR utilities."""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False


class OCRProcessor:
    """Extract text from document images using EasyOCR or Tesseract."""

    def __init__(self, lang: str = "en", use_easy_ocr: bool = True):
        """Initialize OCR processor.
        
        Args:
            lang: Language code ('en' for English)
            use_easy_ocr: Use EasyOCR if True, Tesseract if False
        """
        self.lang = lang
        self.use_easy_ocr = use_easy_ocr
        
        if use_easy_ocr:
            if not EASYOCR_AVAILABLE:
                raise ImportError("easyocr not installed. Install it with: pip install easyocr")
            self.reader = easyocr.Reader([lang], gpu=False)
        else:
            if not PYTESSERACT_AVAILABLE:
                raise ImportError("pytesseract not installed. Install it with: pip install pytesseract")

    def extract_text(self, image_path: str) -> str:
        """Extract text from image file."""
        if self.use_easy_ocr:
            result = self.reader.readtext(image_path)
            text = "\n".join([line[1] for line in result])
            return text.strip()
        else:
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image, lang=self.lang)
            return text.strip()

    def extract_text_with_confidence(self, image_path: str) -> Dict:
        """Extract text and confidence scores."""
        if self.use_easy_ocr:
            result = self.reader.readtext(image_path)
            data = {
                "text": [line[1] for line in result],
                "confidence": [line[2] for line in result],
                "boxes": [line[0] for line in result],
            }
            return data
        else:
            image = Image.open(image_path)
            data = pytesseract.image_to_data(image, lang=self.lang, output_type=pytesseract.Output.DICT)
            return data

    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        image_ext: str = ".png",
    ) -> None:
        """Process all images in a directory."""
        os.makedirs(output_dir, exist_ok=True)
        image_files = list(Path(input_dir).glob(f"*{image_ext}"))
        
        if not image_files:
            print(f"No {image_ext} files found in {input_dir}")
            return

        for idx, image_file in enumerate(image_files, 1):
            try:
                text = self.extract_text(str(image_file))
                output_file = Path(output_dir) / f"{image_file.stem}.txt"
                output_file.write_text(text)
                print(f"[{idx}/{len(image_files)}] Processed {image_file.name}")
            except Exception as e:
                print(f"Error processing {image_file.name}: {e}")


def normalize_text(text: str, max_length: int = 512) -> str:
    """Normalize and clean OCR text."""
    # Remove extra whitespace
    text = " ".join(text.split())
    # Remove non-ASCII characters (optional)
    text = text.encode("ascii", "ignore").decode("ascii")
    # Truncate to max length
    text = text[:max_length]
    return text


def save_ocr_metadata(
    image_path: str,
    text: str,
    label: str,
    output_path: str,
) -> None:
    """Save OCR results and metadata as JSON."""
    metadata = {
        "image_path": image_path,
        "text": text,
        "label": label,
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)


def load_ocr_metadata(path: str) -> Dict:
    """Load OCR metadata from JSON."""
    with open(path) as f:
        return json.load(f)
