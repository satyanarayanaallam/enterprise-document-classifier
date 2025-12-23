# Enterprise Document Classifier - Getting Started

This guide helps you get started with the enterprise document classifier project.

## Setup

### 1. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Data
Download datasets from:
- [RVL-CDIP](https://www.cs.cmu.edu/~aharley/rvl-cdip/) for document classification
- [FUNSD](https://guillaumejaume.github.io/FUNSD/) for layout analysis
- [DocVQA](https://docvqa.cs.st-andrews.ac.uk/) for document QA

Place raw data in `data/raw/`.

## Project Structure

```
project-root/
├─ data/                      # Data directory
│  ├─ raw/                    # Raw datasets
│  ├─ processed/              # Preprocessed data
│  └─ samples/                # Sample data for quick tests
├─ src/                       # Source code
│  ├─ data/                   # Data loading & OCR
│  ├─ models/                 # Model architectures
│  ├─ training/               # Training loops
│  ├─ retrieval/              # FAISS retriever
│  ├─ reader/                 # QA reader
│  ├─ inference/              # FastAPI app & exports
│  └─ utils/                  # Config & logging
├─ notebooks/                 # Jupyter notebooks
├─ deploy/                    # Docker & K8s configs
├─ tests/                     # Unit tests
└─ requirements.txt           # Dependencies
```

## Quick Start

### 1. Prepare Data
```python
from src.data import OCRProcessor

ocr = OCRProcessor()
ocr.process_directory("data/raw/images", "data/processed/ocr_output")
```

### 2. Train Image Classifier
```bash
python -m src.training.train_image_classifier \
  --metadata-dir data/processed/metadata \
  --image-dir data/processed/images \
  --batch-size 32 \
  --num-epochs 10
```

### 3. Train Joint Embeddings
```bash
python -m src.training.train_joint_embeddings \
  --metadata-dir data/processed/metadata \
  --image-dir data/processed/images \
  --batch-size 32 \
  --num-epochs 10
```

### 4. Start Inference Server
```bash
uvicorn src.inference.app:app --reload --port 8000
```

Visit `http://localhost:8000/docs` for interactive API documentation.

## Docker Deployment

### Build Docker Image
```bash
docker build -t document-classifier:latest -f deploy/Dockerfile .
```

### Run with Docker Compose
```bash
docker-compose -f deploy/docker-compose.yml up
```

This starts:
- API server on port 8000
- MLflow tracking server on port 5000

## Kubernetes Deployment

```bash
kubectl apply -f deploy/k8s/deployment.yaml
```

This creates:
- 3 API replicas
- LoadBalancer service
- GPU resource allocation

## Testing

Run tests:
```bash
pytest tests/
```

Run specific test:
```bash
pytest tests/test_models.py::test_image_classifier
```

## Model Export

Export models for production:
```python
from src.inference.export import export_to_torchscript, export_to_onnx

# Export to TorchScript
export_to_torchscript(model, example_input, "models/classifier.pt")

# Export to ONNX
export_to_onnx(model, example_input, "models/classifier.onnx")
```

## API Usage

### Classification
```bash
curl -X POST "http://localhost:8000/classify" \
  -F "file=@document.png"
```

Response:
```json
{
  "predicted_label": "invoice",
  "confidence": 0.95,
  "top_k_labels": [
    ["invoice", 0.95],
    ["receipt", 0.04],
    ["form", 0.01]
  ]
}
```

## Configuration

Modify `src/utils/config.py` for:
- Model architectures
- Training hyperparameters
- Inference settings
- Device configuration

Or set environment variables:
```bash
export DEVICE=cuda
export BATCH_SIZE=64
export LEARNING_RATE=0.0001
export NUM_EPOCHS=20
```

## Monitoring

View training progress with MLflow:
```bash
mlflow ui
```

Then visit `http://localhost:5000` in your browser.

## Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Transformers Library](https://huggingface.co/transformers/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)

## Contributing

1. Create a feature branch
2. Make changes
3. Add tests
4. Submit pull request

## License

See LICENSE file for details.
