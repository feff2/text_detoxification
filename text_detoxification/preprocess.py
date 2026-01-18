"""Data preprocessing module."""

import logging
from typing import Optional

from datasets import Dataset

logger = logging.getLogger(__name__)


def create_alpaca_prompt() -> str:
    """
    Create Alpaca-style prompt template for detoxification.

    Returns:
        Prompt template string
    """
    return """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Rewrite the following toxic or offensive text to make it polite, respectful, and appropriate while preserving the original meaning and intent. Remove all profanity, insults, aggression, and offensive language. Make the tone neutral and constructive.

### Input:
{}

### Response:
{}"""


def format_prompts(
    dataset: Dataset,
    prompt_template: str,
    input_column: str,
    output_column: str,
    eos_token: str,
) -> Dataset:
    """
    Format dataset with prompts for training.

    Args:
        dataset: Input dataset
        prompt_template: Template string for prompts
        input_column: Name of input column
        output_column: Name of output column
        eos_token: End-of-sequence token

    Returns:
        Dataset with formatted text column
    """
    logger.info("Formatting prompts for dataset")

    def formatting_func(examples):
        inputs = examples[input_column]
        outputs = examples[output_column]

        texts = []
        for input_text, output in zip(inputs, outputs):
            if input_text is None or output is None:
                continue
            text = prompt_template.format(input_text, output) + eos_token
            texts.append(text)
        return {"text": texts}

    formatted_dataset = dataset.map(formatting_func, batched=True, remove_columns=dataset.column_names)
    logger.info(f"Formatted {len(formatted_dataset)} examples")
    return formatted_dataset


def split_dataset(
    dataset: Dataset,
    test_size: float = 0.1,
    seed: int = 42,
) -> tuple[Dataset, Dataset]:
    """
    Split dataset into train and validation sets.

    Args:
        dataset: Input dataset
        test_size: Proportion of data for validation
        seed: Random seed

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    logger.info(f"Splitting dataset with test_size={test_size}, seed={seed}")
    split = dataset.train_test_split(test_size=test_size, seed=seed)
    train_dataset = split["train"]
    val_dataset = split["test"]
    logger.info(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    return train_dataset, val_dataset

