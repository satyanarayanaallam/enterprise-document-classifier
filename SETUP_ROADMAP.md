# Setup Roadmap - Enterprise Document Classifier

## Phase 1: Environment Setup (15 minutes)
**Goal:** Get your development environment ready

### Step 1.1: Create Virtual Environment
```bash
cd /Users/satyanarayanaallam/Projects/enterprise-document-classifier
python -m venv venv
source venv/bin/activate
```

**Verify:** You should see `(venv)` in your terminal prompt

### Step 1.2: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Verify:** Run `pip list | grep torch` - should show torch version 2.0+

### Step 1.3: Verify Installation
```bash
python -c "import torch; import transformers; import faiss; print('✓ All packages installed')"
```

**Expected Output:** `✓ All packages installed`

---

## Phase 2: Understand the Project (20 minutes)
**Goal:** Familiarize yourself with the codebase structure

### Step 2.1: Read Documentation
1. Read `README.md` - Project overview (2 min)
2. Read `GETTING_STARTED.md` - Quick reference (3 min)
3. Read `PROJECT_SETUP.md` - Detailed structure (5 min)
4. Read `MODEL_CARD.md` - Model information (5 min)

### Step 2.2: Explore Key Files
```bash
# View the main source structure
ls -la src/
```

Focus on these core modules:
- `src/models/` - Model architectures (ResNet50, DistilBERT)
- `src/data/` - Data loading and OCR
- `src/training/` - Training loops
- `src/inference/` - FastAPI service

### Step 2.3: Run Tests
```bash
# Install pytest if not already installed
pip install pytest

# Run all tests
pytest tests/ -v
```

**Expected:** All tests should pass (green checkmarks)

---

## Phase 3: Prepare Sample Data (20 minutes)
**Goal:** Create minimal sample data to test the pipeline

### Step 3.1: Create Sample Images & Metadata
```bash
# Create directory structure
mkdir -p data/processed/images
mkdir -p data/processed/metadata

# Run the sample data creation script
python -c "
from PIL import Image
import numpy as np
import json
import os

# Create 5 sample document images
for i in range(5):
    # Create random RGB image
    img_array = np.random.randint(100, 200, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    img.save(f'data/processed/images/sample_{i}.png')
    
    # Create metadata
    metadata = {
        'image_path': f'sample_{i}.png',
        'text': f'This is sample document number {i}',
        'label': 'invoice' if i % 2 == 0 else 'contract'
    }
    with open(f'data/processed/metadata/sample_{i}.json', 'w') as f:
        json.dump(metadata, f)

print('✓ Created 5 sample documents')
"
```

**Verify:** Check the created files
```bash
ls -la data/processed/images/
ls -la data/processed/metadata/
```

---

## Phase 4: Test Data Loading (10 minutes)
**Goal:** Verify the data pipeline works

### Step 4.1: Test Dataset
```bash
python -c "
from src.data import DocumentDataset

dataset = DocumentDataset(
    metadata_dir='data/processed/metadata',
    image_dir='data/processed/images',
    image_size=224,
)

print(f'✓ Dataset loaded: {len(dataset)} samples')
print(f'✓ Labels: {dataset.label_to_idx}')

# Test getting a sample
image, text, label = dataset[0]
print(f'✓ Image shape: {image.shape}')
print(f'✓ Text: {text[:50]}...')
print(f'✓ Label index: {label}')
"
```

**Expected Output:** Should show dataset size and sample shapes

---

## Phase 5: Create & Test Models (15 minutes)
**Goal:** Verify models can be instantiated and run inference

### Step 5.1: Test Image Classifier
```bash
python -c "
import torch
from src.models import ImageClassifier

# Create model
model = ImageClassifier(num_classes=10, pretrained=False)
print(f'✓ ImageClassifier created')
print(f'✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}')

# Test inference
x = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    logits = model(x)
    embeddings = model.get_embeddings(x)

print(f'✓ Output shape: {logits.shape}')
print(f'✓ Embedding shape: {embeddings.shape}')
"
```

### Step 5.2: Test Text Encoder
```bash
python -c "
from src.models import TextEncoder

# Create model
model = TextEncoder(model_name='distilbert-base-uncased')
print(f'✓ TextEncoder created')

# Test encoding
texts = ['sample document one', 'sample document two']
embeddings = model.encode_texts(texts)
print(f'✓ Embeddings shape: {embeddings.shape}')
"
```

### Step 5.3: Test Joint Embedder
```bash
python -c "
import torch
from src.models import JointEmbedder

# Create model
model = JointEmbedder(embedding_dim=768)
print(f'✓ JointEmbedder created')

# Test forward pass
images = torch.randn(2, 3, 224, 224)
input_ids = torch.randint(0, 1000, (2, 50))
attention_mask = torch.ones(2, 50)

image_emb, text_emb = model(images, input_ids, attention_mask)
print(f'✓ Image embeddings: {image_emb.shape}')
print(f'✓ Text embeddings: {text_emb.shape}')
"
```

---

## Phase 6: Run Training Demo (30 minutes)
**Goal:** Train a simple model with sample data

### Step 6.1: Start Training
```bash
python src/training/train_image_classifier.py \
    --metadata-dir data/processed/metadata \
    --image-dir data/processed/images \
    --batch-size 2 \
    --num-epochs 2 \
    --device cpu
```

**What to watch for:**
- Training progress bar
- Decreasing loss values
- Validation metrics
- Checkpoint saved messages

**Note:** With only 5 samples and CPU, this will be slow but tests the pipeline

### Step 6.2: Check Output
```bash
# Check if checkpoint was saved
ls -la experiments/checkpoints/
```

---

## Phase 7: Start Inference API (10 minutes)
**Goal:** Get the FastAPI service running

### Step 7.1: Start the Server
```bash
uvicorn src.inference.app:app --reload --port 8000
```

**Expected Output:**
```
Uvicorn running on http://127.0.0.1:8000
```

### Step 7.2: Test the API (in a new terminal)
```bash
# Keep the API server running, open a new terminal window

# Check health
curl http://localhost:8000/health

# Get API info
curl http://localhost:8000/info
```

**Expected Output:**
```json
{"status": "healthy"}
{"name": "Enterprise Document Classifier API", "version": "0.1.0", ...}
```

---

## Phase 8: Run Tests (5 minutes)
**Goal:** Verify all components work together

### Step 8.1: Run Full Test Suite
```bash
pytest tests/ -v --tb=short
```

**Expected:** All tests should pass

### Step 8.2: Run Specific Test
```bash
# Test models specifically
pytest tests/test_models.py -v
```

---

## Phase 9: Docker Setup (Optional, 15 minutes)
**Goal:** Build and test Docker containerization

### Step 9.1: Build Docker Image
```bash
docker build -t document-classifier:latest -f deploy/Dockerfile .
```

### Step 9.2: Run Docker Container
```bash
docker run -p 8000:8000 document-classifier:latest
```

---

## Phase 10: Real Data Integration (When Ready)
**Goal:** Use actual datasets (RVL-CDIP, FUNSD, DocVQA)

### Step 10.1: Download Datasets
1. Visit [RVL-CDIP](https://www.cs.cmu.edu/~aharley/rvl-cdip/)
2. Visit [FUNSD](https://guillaumejaume.github.io/FUNSD/)
3. Visit [DocVQA](https://docvqa.cs.st-andrews.ac.uk/)

### Step 10.2: Extract to data/raw/
```bash
# Organize datasets
data/raw/
├─ rvl-cdip/
├─ funsd/
└─ docvqa/
```

### Step 10.3: Run OCR & Preprocessing
```bash
python -c "
from src.data import OCRProcessor

# Extract text from images
ocr = OCRProcessor()
ocr.process_directory(
    'data/raw/rvl-cdip/images',
    'data/processed/ocr_output'
)
"
```

---

## 📋 Quick Reference Checklist

- [ ] Phase 1: Environment setup (venv + pip install)
- [ ] Phase 2: Read documentation + explore code
- [ ] Phase 3: Run existing tests
- [ ] Phase 4: Create sample data
- [ ] Phase 5: Test data loading pipeline
- [ ] Phase 6: Test models (image, text, joint)
- [ ] Phase 7: Run training on sample data
- [ ] Phase 8: Start FastAPI server
- [ ] Phase 9: Test API endpoints
- [ ] Phase 10: Run full test suite
- [ ] Phase 11: Docker build (optional)
- [ ] Phase 12: Integrate real datasets (later)

---

## ⏱️ Total Time Estimate
- **Quick setup:** ~60 minutes (Phases 1-8)
- **Full setup with tests:** ~90 minutes (all phases)
- **With Docker:** ~105 minutes

## 🆘 Troubleshooting

### Virtual Environment Not Activating
```bash
# Try explicit path
source /Users/satyanarayanaallam/Projects/enterprise-document-classifier/venv/bin/activate
```

### Package Installation Issues
```bash
# Upgrade pip first
python -m pip install --upgrade pip
# Then install requirements
pip install -r requirements.txt --no-cache-dir
```

### CUDA Issues
```bash
# If CUDA unavailable, use CPU
python -c "import torch; print(torch.cuda.is_available())"
# Code will automatically use CPU if CUDA unavailable
```

### Import Errors
```bash
# Verify venv is activated
which python  # Should show path in venv/

# Reinstall package
pip install -e .
```

---

## 🎯 Next: Start Phase 1!

Run this command now to begin:

```bash
cd /Users/satyanarayanaallam/Projects/enterprise-document-classifier
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Then reply with what you see, and we'll continue to Phase 2!
