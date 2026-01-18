"""Data loading module with DVC integration."""

import logging
from pathlib import Path
from typing import Optional

import dvc.api
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_dataset

logger = logging.getLogger(__name__)


def download_data(data_path: str, dvc_remote: Optional[str] = None) -> None:
    """
    Download data from DVC storage.

    Args:
        data_path: Path to data file in DVC
        dvc_remote: Optional DVC remote name
    """
    try:
        from dvc.repo import Repo

        repo = Repo(".")
        repo.pull(targets=[data_path], remote=dvc_remote)
        logger.info(f"Data downloaded from DVC: {data_path}")
    except Exception as e:
        logger.warning(f"Could not download from DVC: {e}. Using local path.")


def load_paradetox_dataset(dataset_name: str) -> Dataset:
    """
    Load multilingual ParaDetox dataset from HuggingFace.

    Args:
        dataset_name: Name of the dataset on HuggingFace

    Returns:
        Dataset containing multilingual detoxification examples
    """
    logger.info(f"Loading {dataset_name} dataset from HuggingFace")
    dataset = load_dataset(dataset_name)
    return dataset


def load_custom_dataset(data_path: str) -> pd.DataFrame:
    """
    Load custom dataset from TSV file.

    Args:
        data_path: Path to TSV file

    Returns:
        DataFrame with custom data
    """
    logger.info(f"Loading custom dataset from {data_path}")
    data_path_obj = Path(data_path)

    if not data_path_obj.exists():
        logger.warning(f"File {data_path} does not exist. Attempting DVC download.")
        download_data(str(data_path))

    df = pd.read_csv(data_path, sep="\t")
    logger.info(f"Loaded {len(df)} examples from custom dataset")
    return df


def combine_datasets(
    paradetox_dataset: Dataset,
    custom_df: Optional[pd.DataFrame],
    languages: list[str],
) -> Dataset:
    """
    Combine ParaDetox dataset with custom dataset.

    Args:
        paradetox_dataset: ParaDetox dataset from HuggingFace
        custom_df: Optional custom DataFrame
        languages: List of language codes to include from ParaDetox

    Returns:
        Combined dataset
    """
    all_datasets = []

    for lang in languages:
        if lang in paradetox_dataset:
            all_datasets.append(paradetox_dataset[lang])
            logger.info(f"Added {lang} dataset: {len(paradetox_dataset[lang])} examples")

    if custom_df is not None:
        custom_dataset = Dataset.from_pandas(custom_df)
        all_datasets.append(custom_dataset)
        logger.info(f"Added custom dataset: {len(custom_df)} examples")

    if not all_datasets:
        raise ValueError("No datasets to combine")

    combined = concatenate_datasets(all_datasets)
    logger.info(f"Combined dataset size: {len(combined)} examples")
    return combined

