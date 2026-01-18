"""Model conversion to ONNX and TensorRT."""

import logging
from pathlib import Path
from typing import Optional

import torch
from unsloth import FastLanguageModel

logger = logging.getLogger(__name__)


def convert_to_onnx(
    model_path: str,
    output_path: str,
    max_seq_length: int = 2048,
    opset_version: int = 14,
) -> None:
    """
    Convert model to ONNX format.

    Args:
        model_path: Path to saved model
        output_path: Path to save ONNX model
        max_seq_length: Maximum sequence length
        opset_version: ONNX opset version
    """
    logger.info(f"Converting model from {model_path} to ONNX format")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=False,
    )

    FastLanguageModel.for_inference(model)

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = tokenizer(
        "Hello, this is a test.",
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_seq_length,
    )

    try:
        torch.onnx.export(
            model,
            (dummy_input["input_ids"], dummy_input.get("attention_mask")),
            str(output_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size", 1: "sequence_length"},
            },
            opset_version=opset_version,
            do_constant_folding=True,
        )
        logger.info(f"ONNX model saved to {output_path}")
    except Exception as e:
        logger.error(f"Error converting to ONNX: {e}")
        logger.warning(
            "ONNX conversion may not work for all model architectures. "
            "Consider using TensorRT conversion instead."
        )
        raise


def convert_to_tensorrt(
    onnx_path: str,
    output_path: str,
    precision: str = "fp16",
    max_batch_size: int = 1,
    max_seq_length: int = 2048,
) -> None:
    """
    Convert ONNX model to TensorRT format.

    Args:
        onnx_path: Path to ONNX model
        output_path: Path to save TensorRT model
        precision: Precision mode (fp32, fp16, int8)
        max_batch_size: Maximum batch size
        max_seq_length: Maximum sequence length
    """
    logger.info(f"Converting ONNX model from {onnx_path} to TensorRT format")

    try:
        import tensorrt as trt

        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)

        with open(onnx_path, "rb") as model_file:
            if not parser.parse(model_file.read()):
                for error in range(parser.num_errors):
                    logger.error(parser.get_error(error))
                raise RuntimeError("Failed to parse ONNX model")

        config = builder.create_builder_config()
        if precision == "fp16":
            config.set_flag(trt.BuilderFlag.FP16)
        elif precision == "int8":
            config.set_flag(trt.BuilderFlag.INT8)

        config.max_workspace_size = 1 << 30

        profile = builder.create_optimization_profile()
        profile.set_shape(
            "input_ids",
            (1, 1),
            (max_batch_size, max_seq_length // 2),
            (max_batch_size, max_seq_length),
        )
        profile.set_shape(
            "attention_mask",
            (1, 1),
            (max_batch_size, max_seq_length // 2),
            (max_batch_size, max_seq_length),
        )
        config.add_optimization_profile(profile)

        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        engine = builder.build_engine(network, config)
        with open(output_path, "wb") as engine_file:
            engine_file.write(engine.serialize())

        logger.info(f"TensorRT model saved to {output_path}")

    except ImportError:
        logger.error(
            "TensorRT is not installed. Please install it to use TensorRT conversion. "
            "You can install it from: https://developer.nvidia.com/tensorrt"
        )
        raise
    except Exception as e:
        logger.error(f"Error converting to TensorRT: {e}")
        raise

