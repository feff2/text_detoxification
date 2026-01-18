"""Model definition and PyTorch Lightning integration."""

import logging
from pathlib import Path
from typing import Optional

import torch
from pytorch_lightning import LightningModule
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    get_linear_schedule_with_warmup,
)
from trl import SFTTrainer
from unsloth import FastLanguageModel

logger = logging.getLogger(__name__)


class DetoxificationModel(LightningModule):
    """PyTorch Lightning module for text detoxification."""

    def __init__(
        self,
        model_name: str,
        max_seq_length: int = 2048,
        dtype: Optional[torch.dtype] = None,
        load_in_4bit: bool = True,
        lora_r: int = 128,
        lora_alpha: int = 256,
        lora_dropout: float = 0.05,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 10,
        gradient_accumulation_steps: int = 2,
    ):
        """
        Initialize detoxification model.

        Args:
            model_name: Name of the base model
            max_seq_length: Maximum sequence length
            dtype: Data type for model
            load_in_4bit: Whether to load model in 4-bit
            lora_r: LoRA rank
            lora_alpha: LoRA alpha parameter
            lora_dropout: LoRA dropout rate
            learning_rate: Learning rate
            weight_decay: Weight decay
            warmup_steps: Number of warmup steps
            gradient_accumulation_steps: Gradient accumulation steps
        """
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps

        logger.info(f"Loading model: {model_name}")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )

        logger.info("Applying LoRA adapters")
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=lora_r,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
            use_rslora=False,
            loftq_config=None,
        )

        self.trainer_obj: Optional[SFTTrainer] = None

    def setup(self, stage: str) -> None:
        """Setup model for training or validation."""
        if stage == "fit":
            logger.info("Setting up model for training")

    def training_step(self, batch, batch_idx):
        """Training step - handled by SFTTrainer."""
        pass

    def validation_step(self, batch, batch_idx):
        """Validation step - handled by SFTTrainer."""
        pass

    def configure_optimizers(self):
        """Configure optimizers."""
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        return optimizer

    def save_model(self, output_dir: str) -> None:
        """
        Save model and tokenizer.

        Args:
            output_dir: Directory to save model
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving model to {output_dir}")
        self.model.save_pretrained(str(output_path))
        self.tokenizer.save_pretrained(str(output_path))

    def prepare_for_inference(self) -> None:
        """Prepare model for inference."""
        logger.info("Preparing model for inference")
        FastLanguageModel.for_inference(self.model)

