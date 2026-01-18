"""Utility functions for plotting and logging."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def plot_training_metrics(
    train_losses: list[float],
    val_losses: list[float],
    output_path: str,
) -> None:
    """
    Plot training and validation losses.

    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        output_path: Path to save the plot
    """
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss", marker="o")
    plt.plot(val_losses, label="Validation Loss", marker="s")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

    logger.info(f"Training metrics plot saved to {output_path}")


def plot_learning_rate_schedule(
    learning_rates: list[float],
    output_path: str,
) -> None:
    """
    Plot learning rate schedule.

    Args:
        learning_rates: List of learning rates
        output_path: Path to save the plot
    """
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(learning_rates, label="Learning Rate", marker="o")
    plt.xlabel("Step")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

    logger.info(f"Learning rate plot saved to {output_path}")


def plot_metric_comparison(
    metrics: dict[str, list[float]],
    output_path: str,
    title: str = "Metrics Comparison",
) -> None:
    """
    Plot multiple metrics for comparison.

    Args:
        metrics: Dictionary of metric names to values
        output_path: Path to save the plot
        title: Plot title
    """
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))
    for metric_name, values in metrics.items():
        plt.plot(values, label=metric_name, marker="o")

    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

    logger.info(f"Metric comparison plot saved to {output_path}")

