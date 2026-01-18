"""Inference module for text detoxification."""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from tqdm import tqdm
from unsloth import FastLanguageModel

from text_detoxification.preprocess import create_alpaca_prompt

logger = logging.getLogger(__name__)


class DetoxificationInference:
    """Inference class for text detoxification."""

    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = 128,
        temperature: float = 0.3,
        top_p: float = 0.9,
    ):
        """
        Initialize inference model.

        Args:
            model_path: Path to saved model
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
        """
        self.model_path = Path(model_path)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        logger.info(f"Loading model from {model_path}")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(self.model_path),
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )

        FastLanguageModel.for_inference(self.model)
        self.prompt_template = create_alpaca_prompt()

        logger.info("Model loaded and ready for inference")

    def detoxify_text(self, toxic_text: str) -> str:
        """
        Detoxify a single text.

        Args:
            toxic_text: Input toxic text

        Returns:
            Detoxified text
        """
        prompt = self.prompt_template.format(toxic_text, "")
        inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                use_cache=True,
            )

        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "### Response:" in result:
            detoxified = result.split("### Response:")[-1].strip()
        else:
            detoxified = result

        return detoxified

    def detoxify_batch(
        self,
        texts: list[str],
        output_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Detoxify a batch of texts.

        Args:
            texts: List of toxic texts
            output_path: Optional path to save results

        Returns:
            DataFrame with results
        """
        logger.info(f"Processing {len(texts)} texts")
        detoxified_texts = []

        for toxic_text in tqdm(texts, desc="Detoxifying"):
            try:
                detox_text = self.detoxify_text(toxic_text)
                detoxified_texts.append(detox_text)
            except Exception as e:
                logger.error(f"Error processing text '{toxic_text[:50]}...': {e}")
                detoxified_texts.append("")

        result_df = pd.DataFrame(
            {
                "id": range(len(texts)),
                "toxic": texts,
                "detoxified": detoxified_texts,
            }
        )

        if output_path:
            output_path_obj = Path(output_path)
            output_path_obj.parent.mkdir(parents=True, exist_ok=True)
            result_df.to_csv(output_path, sep="\t", index=False)
            logger.info(f"Results saved to {output_path}")

        return result_df

    def detoxify_from_file(
        self,
        input_path: str,
        output_path: str,
        text_column: str = "tat_toxic",
        id_column: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Detoxify texts from a TSV file.

        Args:
            input_path: Path to input TSV file
            output_path: Path to save results
            text_column: Name of column with toxic text
            id_column: Optional name of ID column

        Returns:
            DataFrame with results
        """
        logger.info(f"Loading data from {input_path}")
        df = pd.read_csv(input_path, sep="\t")

        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in input file")

        texts = df[text_column].tolist()
        detoxified_texts = []

        for toxic_text in tqdm(texts, desc="Detoxifying"):
            try:
                detox_text = self.detoxify_text(toxic_text)
                detoxified_texts.append(detox_text)
            except Exception as e:
                logger.error(f"Error processing text '{toxic_text[:50]}...': {e}")
                detoxified_texts.append("")

        result_df = pd.DataFrame(
            {
                "id": df[id_column].tolist() if id_column and id_column in df.columns else range(len(df)),
                text_column: texts,
                "tat_detox1": detoxified_texts,
            }
        )

        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(output_path, sep="\t", index=False)
        logger.info(f"Results saved to {output_path}")

        return result_df

