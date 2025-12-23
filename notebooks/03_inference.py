"""
Inference & Deployment Notebook

This notebook demonstrates:
1. Loading trained models
2. Running inference on documents
3. Using the retrieval system
4. Exporting models for deployment
"""

import torch
from src.models import ImageClassifier
from src.inference.export import export_to_torchscript, export_to_onnx
from src.utils import Config, setup_logger

logger = setup_logger(__name__)


def main():
    """Run inference demo."""
    config = Config()

    # Load model
    try:
        model = ImageClassifier(num_classes=10)
        checkpoint_path = f"{config.model_checkpoint_dir}/best_model.pt"
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded model from {checkpoint_path}")
    except FileNotFoundError:
        logger.warning("Model checkpoint not found. Using untrained model.")
        model = ImageClassifier(num_classes=10)

    model.eval()

    # Example inference (requires actual image)
    try:
        from PIL import Image
        import torchvision.transforms as transforms

        # Load and preprocess image
        image = Image.open("sample_document.png")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        image_tensor = transform(image).unsqueeze(0)

        # Inference
        with torch.no_grad():
            logits = model(image_tensor)
            probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1)
            confidence = probs[0, pred_class].item()

        logger.info(f"Predicted class: {pred_class.item()}, Confidence: {confidence:.4f}")

    except FileNotFoundError:
        logger.info("Sample image not found. Skipping inference demo.")

    # Export models
    logger.info("Exporting models...")
    try:
        example_input = torch.randn(1, 3, 224, 224)

        # Export to TorchScript
        export_to_torchscript(
            model,
            example_input,
            f"{config.model_checkpoint_dir}/classifier.pt",
        )

        # Export to ONNX
        export_to_onnx(
            model,
            example_input,
            f"{config.model_checkpoint_dir}/classifier.onnx",
        )

        logger.info("Models exported successfully!")
    except Exception as e:
        logger.error(f"Export failed: {e}")


if __name__ == "__main__":
    main()
