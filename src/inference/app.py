"""FastAPI inference service."""

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import torch
import io
from PIL import Image

from src.models import ImageClassifier, JointEmbedder
from src.reader import TransformerReader
from src.retrieval import FAISSRetriever
from src.utils import setup_logger, Config

logger = setup_logger(__name__)

app = FastAPI(title="Enterprise Document Classifier API", version="0.1.0")


class DocumentClassificationRequest(BaseModel):
    """Request model for classification."""
    image_path: str
    text: Optional[str] = None


class DocumentClassificationResponse(BaseModel):
    """Response model for classification."""
    predicted_label: str
    confidence: float
    top_k_labels: List[tuple]


class RAGRequest(BaseModel):
    """Request model for RAG."""
    query: str
    num_results: int = 5


class RAGResponse(BaseModel):
    """Response model for RAG."""
    answer: str
    context: str
    confidence: float


# Global models (loaded at startup)
config: Optional[Config] = None
classifier: Optional[ImageClassifier] = None
joint_embedder: Optional[JointEmbedder] = None
reader: Optional[TransformerReader] = None
retriever: Optional[FAISSRetriever] = None
device: str = "cuda" if torch.cuda.is_available() else "cpu"


@app.on_event("startup")
async def load_models():
    """Load models on startup."""
    global config, classifier, joint_embedder, reader, retriever

    logger.info("Loading models...")
    config = Config()

    # Load classification model
    classifier = ImageClassifier(num_classes=10).to(device)
    logger.info("Loaded image classifier")

    # Load joint embedder
    joint_embedder = JointEmbedder(embedding_dim=768).to(device)
    logger.info("Loaded joint embedder")

    # Load reader
    reader = TransformerReader()
    logger.info("Loaded transformer reader")

    # Initialize retriever
    retriever = FAISSRetriever(embedding_dim=768, device=device)
    logger.info("Initialized FAISS retriever")


@app.post("/classify", response_model=DocumentClassificationResponse)
async def classify_document(file: UploadFile = File(...)):
    """Classify a document image.

    Args:
        file: Uploaded document image

    Returns:
        Classification results
    """
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Preprocess
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Classify
        with torch.no_grad():
            logits = classifier(image_tensor)
            probs = torch.softmax(logits, dim=1)
            pred_label_idx = torch.argmax(probs, dim=1)
            confidence = probs[0, pred_label_idx].item()

        # Map to label
        label_map = {i: f"class_{i}" for i in range(10)}
        predicted_label = label_map[pred_label_idx.item()]

        # Get top-k
        top_k_probs, top_k_indices = torch.topk(probs[0], k=3)
        top_k_labels = [
            (label_map[idx.item()], prob.item())
            for prob, idx in zip(top_k_probs, top_k_indices)
        ]

        return DocumentClassificationResponse(
            predicted_label=predicted_label,
            confidence=confidence,
            top_k_labels=top_k_labels,
        )

    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/info")
async def info():
    """Get API info."""
    return {
        "name": "Enterprise Document Classifier API",
        "version": "0.1.0",
        "device": device,
        "models_loaded": classifier is not None,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
