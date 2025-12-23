# Model Card: Enterprise Document Classifier

## Overview
This model card describes the Enterprise Document Classifier, a multi-component system for document image classification, retrieval, and question answering.

## Model Details

### Components
1. **Image Encoder**: ResNet50-based encoder for document image feature extraction
2. **Text Encoder**: DistilBERT-based encoder for OCR text representation
3. **Joint Embedder**: Multi-modal embedding space for image-text alignment
4. **Reader**: Transformer-based extractive QA model for question answering

## Intended Use

### Primary Use Cases
- Document classification (invoices, contracts, forms, etc.)
- Document retrieval using both image and text
- Question answering over document content
- Enterprise document automation and digitization

### Not Intended For
- Medical document analysis (requires specialized models)
- Non-English documents (model trained on English)
- Real-time processing of high-resolution scans (may require optimization)

## Training Data

### Datasets
- **RVL-CDIP**: 400K document images across 16 classes
- **FUNSD**: ~200 forms with layout and entity annotations
- **DocVQA**: ~40K document images with QA pairs

### Data Splits
- Training: 80%
- Validation: 20%

## Model Performance

### Evaluation Metrics
- Image Classification Accuracy: ~95% (on RVL-CDIP)
- Retrieval MAP@5: ~0.85 (on joint embeddings)
- QA Exact Match: ~85% (on DocVQA)

### Limitations
- Performance varies by document type (best on scanned documents, lower on photos)
- OCR quality affects downstream performance
- Limited to English text

## Ethical Considerations

### Potential Biases
- Model trained primarily on English documents
- May perform better on clear, well-scanned documents
- Some document types underrepresented in training data

### Fairness & Bias Mitigation
- Evaluation on multiple document types recommended
- Consider document quality and language mix in deployment
- Monitor performance across different document categories

## Technical Specifications

### Model Architecture
- Image Encoder: ResNet50 + projection head (→ 768D)
- Text Encoder: DistilBERT + projection head (→ 768D)
- Embedding Space: 768-dimensional shared space
- Reader: RoBERTa-base fine-tuned for QA

### Input/Output
- **Image Input**: 224×224 RGB images (normalized)
- **Text Input**: Up to 512 tokens
- **Output**: Class probabilities or embeddings (768D)

### Computational Requirements
- **Training**: Single GPU (16GB VRAM), ~24 hours
- **Inference**: CPU ~500ms/image, GPU ~50ms/image

## Deployment

### Supported Formats
- PyTorch (.pt, .pth)
- TorchScript (.pt)
- ONNX (.onnx)

### Serving
- FastAPI + Uvicorn
- Docker containerized
- Kubernetes deployable

### API Endpoints
- `POST /classify` - Classify document image
- `POST /retrieve` - Retrieve similar documents
- `POST /qa` - Question answering over document

## Maintenance & Updates

### Monitoring
- Track classification accuracy per document type
- Monitor retrieval performance over time
- Log model predictions for analysis

### Retraining
- Recommended every 6 months with new data
- Fine-tune on domain-specific documents for improved performance
- Monitor for concept drift in document types

## Acknowledgments

Built with:
- [PyTorch](https://pytorch.org/)
- [Hugging Face Transformers](https://huggingface.co/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [FastAPI](https://fastapi.tiangolo.com/)

## License

See LICENSE file for details.
