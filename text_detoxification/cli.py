"""CLI module for text detoxification."""

import logging
from pathlib import Path

import fire
import hydra
from dotenv import load_dotenv
from omegaconf import OmegaConf

from text_detoxification import data_loader, inference, preprocess, trainer
from text_detoxification.convert import convert_to_onnx, convert_to_tensorrt
from text_detoxification.model import DetoxificationModel
from text_detoxification.triton_server import run_triton_server

# Load environment variables
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def train(config_path: str = "configs/train.yaml") -> None:
    """
    Train the detoxification model.

    Args:
        config_path: Path to training configuration file
    """
    with hydra.initialize(config_path="../configs", version_base=None):
        cfg = hydra.compose(config_name="train")

    logger.info("Starting training pipeline")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    model_config = cfg.model
    data_config = cfg.data
    train_config = cfg.training

    model = DetoxificationModel(
        model_name=model_config.model_name,
        max_seq_length=model_config.max_seq_length,
        dtype=None,
        load_in_4bit=model_config.load_in_4bit,
        lora_r=model_config.lora_r,
        lora_alpha=model_config.lora_alpha,
        lora_dropout=model_config.lora_dropout,
        learning_rate=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        warmup_steps=train_config.warmup_steps,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
    )

    paradetox_dataset = data_loader.load_paradetox_dataset(data_config.dataset_name)

    custom_df = None
    if data_config.custom_data_path:
        custom_df = data_loader.load_custom_dataset(data_config.custom_data_path)

    combined_dataset = data_loader.combine_datasets(
        paradetox_dataset,
        custom_df,
        data_config.languages,
    )

    prompt_template = preprocess.create_alpaca_prompt()
    formatted_dataset = preprocess.format_prompts(
        combined_dataset,
        prompt_template,
        data_config.input_column,
        data_config.output_column,
        model.tokenizer.eos_token,
    )

    train_dataset, val_dataset = preprocess.split_dataset(
        formatted_dataset,
        test_size=data_config.test_size,
        seed=train_config.seed,
    )

    trainer.train_model(
        model,
        train_dataset,
        val_dataset,
        OmegaConf.to_container(train_config, resolve=True),
        train_config.output_dir,
        cfg.mlflow.uri,
        cfg.mlflow.experiment_name,
    )

    logger.info("Training completed successfully")


def infer(
    model_path: str,
    input_path: str,
    output_path: str,
    text_column: str = "tat_toxic",
    id_column: str = "id",
    max_new_tokens: int = 128,
    temperature: float = 0.3,
    top_p: float = 0.9,
) -> None:
    """Run inference on input data."""
    logger.info("Starting inference")

    inferencer = inference.DetoxificationInference(
        model_path=model_path,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    inferencer.detoxify_from_file(
        input_path=input_path,
        output_path=output_path,
        text_column=text_column,
        id_column=id_column,
    )

    logger.info("Inference completed successfully")


def convert(
    model_path: str,
    output_path: str,
    format: str = "onnx",
    onnx_opset_version: int = 14,
    tensorrt_precision: str = "fp16",
) -> None:
    """Convert model to ONNX or TensorRT format."""
    logger.info(f"Converting model to {format.upper()} format")

    if format.lower() == "onnx":
        convert_to_onnx(
            model_path=model_path,
            output_path=output_path,
            opset_version=onnx_opset_version,
        )
    elif format.lower() == "tensorrt":
        onnx_path = output_path.replace(".trt", ".onnx")
        convert_to_onnx(
            model_path=model_path,
            output_path=onnx_path,
            opset_version=onnx_opset_version,
        )
        convert_to_tensorrt(
            onnx_path=onnx_path,
            output_path=output_path,
            precision=tensorrt_precision,
        )
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'onnx' or 'tensorrt'")

    logger.info("Conversion completed successfully")


def serve(
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
    """Start Triton Inference Server."""
    run_triton_server(
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


def main():
    """Main CLI entry point."""
    fire.Fire(
        {
            "train": train,
            "infer": infer,
            "convert": convert,
            "serve": serve,
        }
    )


if __name__ == "__main__":
    main()
