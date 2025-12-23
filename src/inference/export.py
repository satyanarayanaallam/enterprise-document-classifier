"""Model export utilities."""

import torch
import torch.onnx
from typing import Optional
import os


def export_to_torchscript(
    model: torch.nn.Module,
    example_input: torch.Tensor,
    output_path: str,
) -> None:
    """Export model to TorchScript.

    Args:
        model: PyTorch model
        example_input: Example input for tracing
        output_path: Output file path
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.eval()
    traced = torch.jit.trace(model, example_input)
    torch.jit.save(traced, output_path)
    print(f"Exported to TorchScript: {output_path}")


def export_to_onnx(
    model: torch.nn.Module,
    example_input: torch.Tensor,
    output_path: str,
    input_names: list = None,
    output_names: list = None,
    dynamic_axes: dict = None,
) -> None:
    """Export model to ONNX.

    Args:
        model: PyTorch model
        example_input: Example input
        output_path: Output file path
        input_names: Input names
        output_names: Output names
        dynamic_axes: Dynamic axes for batch dimension
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.eval()

    if input_names is None:
        input_names = ["input"]
    if output_names is None:
        output_names = ["output"]

    torch.onnx.export(
        model,
        example_input,
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=14,
        verbose=False,
    )
    print(f"Exported to ONNX: {output_path}")


def load_torchscript_model(model_path: str, device: str = "cpu"):
    """Load TorchScript model.

    Args:
        model_path: Path to model
        device: Device to load on

    Returns:
        Loaded model
    """
    return torch.jit.load(model_path, map_location=device)
