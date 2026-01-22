"""
ONNX export utilities for converting PyTorch models to ONNX format.

This module provides functionality to export trained PyTorch segmentation
models to ONNX format for deployment with ONNX Runtime.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import segmentation_models_pytorch as smp

logger = logging.getLogger(__name__)

# Default model config for backward compatibility with old checkpoints
DEFAULT_MODEL_CONFIG = {
    "encoder_name": "resnet18",
    "in_channels": 1,
    "classes": 1,
}


def load_pytorch_model(
    checkpoint_path: Path,
    device: torch.device = torch.device("cpu"),
) -> torch.nn.Module:
    """
    Load a trained PyTorch model from checkpoint.

    The model architecture is read from the checkpoint's model_config field.
    For backward compatibility with older checkpoints that don't have this field,
    falls back to default configuration (resnet18 encoder, 1 input channel, 1 class).

    Args:
        checkpoint_path: Path to model checkpoint (.pth file)
        device: Device to load model onto

    Returns:
        Loaded model in eval mode
    """
    logger.info(f"Loading PyTorch model from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Read model config from checkpoint, fall back to defaults for old checkpoints
    model_config = checkpoint.get("model_config", None)
    if model_config is None:
        logger.warning(
            "No model_config in checkpoint, using defaults (resnet18 encoder)"
        )
        model_config = DEFAULT_MODEL_CONFIG
    else:
        logger.info(
            f"Using model config from checkpoint: encoder={model_config.get('encoder_name')}"
        )

    # Create model with architecture from checkpoint
    model = smp.Unet(
        encoder_name=model_config["encoder_name"],
        encoder_weights=None,  # Don't load pretrained weights, we have trained weights
        in_channels=model_config["in_channels"],
        classes=model_config["classes"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    epoch = checkpoint.get("epoch", "unknown")
    logger.info(f"Model loaded successfully (epoch {epoch})")

    return model


def export_to_onnx(
    checkpoint_path: Path,
    output_path: Path,
    input_size: Tuple[int, int] = (256, 256),
    opset_version: int = 17,
    validate: bool = True,
) -> Path:
    """
    Export a PyTorch model checkpoint to ONNX format.

    Args:
        checkpoint_path: Path to PyTorch checkpoint (.pth file)
        output_path: Path for output ONNX model (.onnx file)
        input_size: Input image size (height, width) for tracing
        opset_version: ONNX opset version (default: 17)
        validate: Whether to validate the exported model

    Returns:
        Path to exported ONNX model
    """
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load model
    device = torch.device("cpu")  # Export on CPU for compatibility
    model = load_pytorch_model(checkpoint_path, device)

    # Create dummy input for tracing
    # Shape: [batch, channels, height, width]
    dummy_input = torch.randn(1, 1, input_size[0], input_size[1], device=device)

    logger.info(f"Exporting to ONNX with input shape {dummy_input.shape}")

    # Export to ONNX using legacy exporter (doesn't require onnxscript)
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 2: "height", 3: "width"},
            "output": {0: "batch_size", 2: "height", 3: "width"},
        },
        dynamo=False,  # Use legacy exporter (no onnxscript dependency)
    )

    logger.info(f"ONNX model exported to {output_path}")

    # Validate if requested
    if validate:
        validate_onnx_export(checkpoint_path, output_path, input_size)

    return output_path


def validate_onnx_export(
    checkpoint_path: Path,
    onnx_path: Path,
    input_size: Tuple[int, int] = (256, 256),
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> bool:
    """
    Validate that ONNX model produces same output as PyTorch model.

    Args:
        checkpoint_path: Path to original PyTorch checkpoint
        onnx_path: Path to exported ONNX model
        input_size: Input size for test
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison

    Returns:
        True if validation passes
    """
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        logger.warning("ONNX Runtime not installed, skipping validation")
        return True

    logger.info("Validating ONNX export...")

    # Load ONNX model and check
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX model structure is valid")

    # Load PyTorch model
    device = torch.device("cpu")
    pytorch_model = load_pytorch_model(checkpoint_path, device)

    # Create test input
    test_input = torch.randn(1, 1, input_size[0], input_size[1], device=device)

    # PyTorch inference
    with torch.no_grad():
        pytorch_output = pytorch_model(test_input)
        pytorch_output = pytorch_output.numpy()

    # ONNX Runtime inference
    ort_session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    ort_inputs = {"input": test_input.numpy()}
    ort_output = ort_session.run(None, ort_inputs)[0]

    # Compare outputs
    if np.allclose(pytorch_output, ort_output, rtol=rtol, atol=atol):
        max_diff = np.abs(pytorch_output - ort_output).max()
        logger.info(f"Validation PASSED (max diff: {max_diff:.2e})")
        return True
    else:
        max_diff = np.abs(pytorch_output - ort_output).max()
        mean_diff = np.abs(pytorch_output - ort_output).mean()
        logger.error(
            f"Validation FAILED: max diff = {max_diff:.2e}, mean diff = {mean_diff:.2e}"
        )
        return False


def get_onnx_model_info(onnx_path: Path) -> dict:
    """
    Get information about an ONNX model.

    Args:
        onnx_path: Path to ONNX model

    Returns:
        Dictionary with model information
    """
    try:
        import onnx
    except ImportError:
        return {"error": "ONNX not installed"}

    onnx_path = Path(onnx_path)

    if not onnx_path.exists():
        return {"error": f"File not found: {onnx_path}"}

    model = onnx.load(str(onnx_path))

    # Get input info
    inputs = []
    for inp in model.graph.input:
        shape = [d.dim_value if d.dim_value else d.dim_param for d in inp.type.tensor_type.shape.dim]
        inputs.append({
            "name": inp.name,
            "shape": shape,
            "dtype": inp.type.tensor_type.elem_type,
        })

    # Get output info
    outputs = []
    for out in model.graph.output:
        shape = [d.dim_value if d.dim_value else d.dim_param for d in out.type.tensor_type.shape.dim]
        outputs.append({
            "name": out.name,
            "shape": shape,
            "dtype": out.type.tensor_type.elem_type,
        })

    # File size
    file_size_mb = onnx_path.stat().st_size / (1024 * 1024)

    return {
        "path": str(onnx_path),
        "file_size_mb": round(file_size_mb, 2),
        "opset_version": model.opset_import[0].version if model.opset_import else None,
        "producer": model.producer_name,
        "inputs": inputs,
        "outputs": outputs,
        "num_nodes": len(model.graph.node),
    }
