"""Triton Inference Server module using PyTriton with TensorRT."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton

from text_detoxification.preprocess import create_alpaca_prompt

logger = logging.getLogger(__name__)


class TensorRTInferenceModel:
    """TensorRT model wrapper for Triton Inference Server."""

    def __init__(
        self,
        tensorrt_model_path: str,
        tokenizer_path: str,
        max_seq_length: int = 2048,
        max_new_tokens: int = 128,
        temperature: float = 0.3,
        top_p: float = 0.9,
    ):
        """
        Initialize TensorRT inference model.

        Args:
            tensorrt_model_path: Path to TensorRT engine file
            tokenizer_path: Path to tokenizer
            max_seq_length: Maximum sequence length
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
        """
        self.tensorrt_model_path = Path(tensorrt_model_path)
        self.tokenizer_path = Path(tokenizer_path)
        self.max_seq_length = max_seq_length
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        logger.info(f"Loading TensorRT model from {tensorrt_model_path}")
        logger.info(f"Loading tokenizer from {tokenizer_path}")

        try:
            import tensorrt as trt
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(str(self.tokenizer_path))
            self.prompt_template = create_alpaca_prompt()

            trt_logger = trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(trt_logger)

            with open(self.tensorrt_model_path, "rb") as engine_file:
                engine_data = engine_file.read()

            self.engine = runtime.deserialize_cuda_engine(engine_data)
            if self.engine is None:
                raise RuntimeError("Failed to deserialize TensorRT engine")
            self.context = self.engine.create_execution_context()

            logger.info("TensorRT model loaded successfully")

        except ImportError as e:
            logger.error(f"Required libraries not installed: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading TensorRT model: {e}")
            raise

    def infer(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """
        Run inference using TensorRT engine.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask

        Returns:
            Generated token IDs (logits)
        """
        try:
            import pycuda.driver as cuda
            import pycuda.autoinit

            batch_size = input_ids.shape[0]
            seq_length = input_ids.shape[1]

            input_ids_int = input_ids.astype(np.int32)
            attention_mask_int = attention_mask.astype(np.int32)

            d_input_ids = cuda.mem_alloc(input_ids_int.nbytes)
            d_attention_mask = cuda.mem_alloc(attention_mask_int.nbytes)

            output_shape = self.engine.get_binding_shape(2)
            if output_shape[0] == -1:
                output_shape = (batch_size, seq_length, self.engine.get_binding_shape(2)[-1])
            d_output = cuda.mem_alloc(np.prod(output_shape) * 4)

            cuda.memcpy_htod(d_input_ids, input_ids_int)
            cuda.memcpy_htod(d_attention_mask, attention_mask_int)

            bindings = [int(d_input_ids), int(d_attention_mask), int(d_output)]

            stream = cuda.Stream()
            self.context.execute_async_v2(bindings, stream.handle)
            stream.synchronize()

            output = np.empty(output_shape, dtype=np.float32)
            cuda.memcpy_dtoh(output, d_output)

            return output

        except Exception as e:
            logger.error(f"Error during TensorRT inference: {e}")
            raise


def create_triton_server(
    tensorrt_model_path: str,
    tokenizer_path: str,
    model_name: str = "text_detoxification",
    max_seq_length: int = 2048,
    max_new_tokens: int = 128,
    temperature: float = 0.3,
    top_p: float = 0.9,
    host: str = "0.0.0.0",
    port: int = 8000,
) -> None:
    """
    Create and run Triton Inference Server with TensorRT model.

    Args:
        tensorrt_model_path: Path to TensorRT engine file
        tokenizer_path: Path to tokenizer
        model_name: Name of the model in Triton
        max_seq_length: Maximum sequence length
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        host: Host to bind the server
        port: Port to bind the server
    """
    logger.info("Initializing Triton Inference Server")

    inference_model = TensorRTInferenceModel(
        tensorrt_model_path=tensorrt_model_path,
        tokenizer_path=tokenizer_path,
        max_seq_length=max_seq_length,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    @batch
    def _infer_fn(text_batch: np.ndarray) -> np.ndarray:
        """
        Inference function for Triton.

        Args:
            text_batch: Batch of input texts as numpy array of strings

        Returns:
            Batch of detoxified texts
        """
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(inference_model.tokenizer_path))
        results = []

        for text in text_batch:
            if isinstance(text, bytes):
                text = text.decode("utf-8")
            elif isinstance(text, np.ndarray):
                text = str(text.item())

            prompt = inference_model.prompt_template.format(text, "")
            inputs = tokenizer(
                prompt,
                return_tensors="np",
                padding=True,
                truncation=True,
                max_length=max_seq_length,
            )

            input_ids = inputs["input_ids"].astype(np.int32)
            attention_mask = inputs["attention_mask"].astype(np.int32)

            logits = inference_model.infer(input_ids, attention_mask)

            import torch
            from torch import nn

            current_ids = input_ids[0].copy()
            generated_tokens = []

            for _ in range(inference_model.max_new_tokens):
                logits_batch = inference_model.infer(
                    current_ids.reshape(1, -1),
                    np.ones((1, len(current_ids)), dtype=np.int32),
                )

                next_token_logits = logits_batch[0, -1, :]
                probs = nn.functional.softmax(
                    torch.from_numpy(next_token_logits) / inference_model.temperature, dim=-1
                )

                if inference_model.top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                    sorted_indices_to_remove = cumsum_probs > inference_model.top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = False
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    probs[indices_to_remove] = 0
                    probs = probs / probs.sum()

                next_token_id = torch.multinomial(probs, num_samples=1).item()
                generated_tokens.append(next_token_id)
                current_ids = np.append(current_ids, next_token_id)

                if next_token_id == tokenizer.eos_token_id:
                    break

                if len(current_ids) >= inference_model.max_seq_length:
                    break

            generated_ids = np.concatenate([input_ids[0], generated_tokens])
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

            if "### Response:" in generated_text:
                detoxified = generated_text.split("### Response:")[-1].strip()
            else:
                detoxified = generated_text

            results.append(detoxified.encode("utf-8"))

        return np.array(results, dtype=object)

    with Triton() as triton:
        logger.info(f"Binding model '{model_name}' to Triton server")

        triton.bind(
            model_name=model_name,
            infer_func=_infer_fn,
            inputs=[
                Tensor(name="text", dtype=np.bytes_, shape=(-1,)),
            ],
            outputs=[
                Tensor(name="detoxified_text", dtype=np.bytes_, shape=(-1,)),
            ],
            config=ModelConfig(max_batch_size=8),
        )

        logger.info(f"Starting Triton Inference Server on {host}:{port}")
        triton.serve(host=host, port=port)


def run_triton_server(
    tensorrt_model_path: str,
    tokenizer_path: str,
    model_name: str = "text_detoxification",
    max_seq_length: int = 2048,
    max_new_tokens: int = 128,
    temperature: float = 0.3,
    top_p: float = 0.9,
    host: str = "0.0.0.0",
    port: int = 8000,
) -> None:
    """
    Run Triton Inference Server (wrapper function).

    Args:
        tensorrt_model_path: Path to TensorRT engine file
        tokenizer_path: Path to tokenizer
        model_name: Name of the model in Triton
        max_seq_length: Maximum sequence length
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        host: Host to bind the server
        port: Port to bind the server
    """
    create_triton_server(
        tensorrt_model_path=tensorrt_model_path,
        tokenizer_path=tokenizer_path,
        model_name=model_name,
        max_seq_length=max_seq_length,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        host=host,
        port=port,
    )

