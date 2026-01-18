"""Training module with PyTorch Lightning and MLflow integration."""

import logging
import subprocess
from pathlib import Path
from typing import Optional

from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from transformers import TrainingArguments

from text_detoxification.model import DetoxificationModel

logger = logging.getLogger(__name__)


def get_git_commit_id() -> str:
    """
    Get current git commit ID.

    Returns:
        Git commit hash or 'unknown'
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def setup_mlflow(
    mlflow_uri: str,
    experiment_name: str,
    run_name: Optional[str] = None,
) -> MLFlowLogger:
    """
    Setup MLflow logging.

    Args:
        mlflow_uri: MLflow server URI
        experiment_name: Name of the experiment
        run_name: Optional run name

    Returns:
        MLFlowLogger instance
    """
    logger.info(f"MLflow tracking URI: {mlflow_uri}")
    logger.info(f"Experiment: {experiment_name}")

    mlflow_logger = MLFlowLogger(
        experiment_name=experiment_name,
        tracking_uri=mlflow_uri,
        run_name=run_name,
    )

    return mlflow_logger


def train_model(
    model: DetoxificationModel,
    train_dataset,
    val_dataset,
    config: dict,
    output_dir: str,
    mlflow_uri: str = "http://127.0.0.1:8080",
    experiment_name: str = "text-detoxification",
) -> None:
    """
    Train the detoxification model.

    Args:
        model: DetoxificationModel instance
        train_dataset: Training dataset
        val_dataset: Validation dataset
        config: Training configuration
        output_dir: Output directory for checkpoints
        mlflow_uri: MLflow server URI
        experiment_name: MLflow experiment name
    """
    seed_everything(config.get("seed", 42), workers=True)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    mlflow_logger = setup_mlflow(mlflow_uri, experiment_name)

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(output_path / "checkpoints"),
        monitor="val_loss",
        mode="min",
        save_top_k=2,
        filename="checkpoint-{epoch:02d}-{val_loss:.2f}",
    )

    training_args = TrainingArguments(
        per_device_train_batch_size=config.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=config.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 2),
        warmup_steps=config.get("warmup_steps", 10),
        num_train_epochs=config.get("num_train_epochs", 5),
        learning_rate=config.get("learning_rate", 2e-4),
        fp16=config.get("fp16", True),
        logging_steps=config.get("logging_steps", 10),
        eval_strategy=config.get("eval_strategy", "steps"),
        eval_steps=config.get("eval_steps", 50),
        save_strategy=config.get("save_strategy", "steps"),
        save_steps=config.get("save_steps", 50),
        save_total_limit=config.get("save_total_limit", 2),
        load_best_model_at_end=config.get("load_best_model_at_end", True),
        metric_for_best_model=config.get("metric_for_best_model", "eval_loss"),
        optim=config.get("optim", "adamw_8bit"),
        weight_decay=config.get("weight_decay", 0.01),
        lr_scheduler_type=config.get("lr_scheduler_type", "linear"),
        seed=config.get("seed", 42),
        output_dir=str(output_path),
        report_to="none",
    )

    from trl import SFTTrainer

    trainer_obj = SFTTrainer(
        model=model.model,
        tokenizer=model.tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=config.get("max_seq_length", 2048),
        dataset_num_proc=config.get("dataset_num_proc", 2),
        packing=config.get("packing", False),
        args=training_args,
    )

    model.trainer_obj = trainer_obj

    logger.info("Starting training")
    trainer_stats = trainer_obj.train()

    logger.info("Training completed")

    git_commit = get_git_commit_id()
    mlflow_logger.log_hyperparams({"git_commit": git_commit})
    mlflow_logger.log_hyperparams(config)

    metrics_to_log = {"train_loss": trainer_stats.training_loss}

    if hasattr(trainer_obj.state, "log_history"):
        train_losses = []
        eval_losses = []
        for log_entry in trainer_obj.state.log_history:
            if isinstance(log_entry, dict):
                for key, value in log_entry.items():
                    if isinstance(value, (int, float)):
                        if "train_loss" in key.lower():
                            train_losses.append(value)
                            metrics_to_log[key] = value
                        elif "eval_loss" in key.lower() or "loss" in key.lower():
                            eval_losses.append(value)
                            metrics_to_log[key] = value
                        elif "metric" in key.lower() or "eval" in key.lower():
                            metrics_to_log[key] = value

    mlflow_logger.log_metrics(metrics_to_log)

    model_path = output_path / "final_model"
    model.save_model(str(model_path))

    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    if len(metrics_to_log) >= 3:
        from text_detoxification.utils import (
            plot_learning_rate_schedule,
            plot_metric_comparison,
            plot_training_metrics,
        )

        train_losses = [
            v for k, v in metrics_to_log.items() if "train" in k.lower() and "loss" in k.lower()
        ]
        val_losses = [
            v for k, v in metrics_to_log.items() if "eval" in k.lower() or "val" in k.lower()
        ]
        lrs = [v for k, v in metrics_to_log.items() if "learning_rate" in k.lower()]

        if train_losses:
            p_train = plots_dir / "train_loss.png"
            plot_metric_comparison({"Train Loss": train_losses}, str(p_train), "Train Loss")
            mlflow_logger.experiment.log_artifact(mlflow_logger.run_id, str(p_train), "plots")

        if val_losses:
            p_val = plots_dir / "val_loss.png"
            plot_metric_comparison({"Val Loss": val_losses}, str(p_val), "Validation Loss")
            mlflow_logger.experiment.log_artifact(mlflow_logger.run_id, str(p_val), "plots")

        if lrs:
            p_lr = plots_dir / "lr_schedule.png"
            plot_learning_rate_schedule(lrs, str(p_lr))
            mlflow_logger.experiment.log_artifact(mlflow_logger.run_id, str(p_lr), "plots")

        # Keeping the summary plot as well
        train_metrics = {k: [v] for k, v in metrics_to_log.items() if "train" in k.lower()}
        val_metrics = {
            k: [v] for k, v in metrics_to_log.items() if "eval" in k.lower() or "val" in k.lower()
        }

        if train_metrics and val_metrics:
            all_metrics = {**train_metrics, **val_metrics}
            plot_path = plots_dir / "metrics_summary.png"
            plot_metric_comparison(all_metrics, str(plot_path), "Metrics Summary")
            mlflow_logger.experiment.log_artifact(mlflow_logger.run_id, str(plot_path), "plots")

    logger.info(f"Model saved to {model_path}")

