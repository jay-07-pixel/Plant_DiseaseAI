#!/usr/bin/env python3
"""Train EfficientNet-B0, evaluate, export, and verify production models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluator import Evaluator
from exports.exporter import ModelExporter
from inference.predictor import Predictor
from training.trainer import Trainer
from utils.config import AppConfig, load_config
from utils.device import get_device
from utils.logging import setup_logging
from utils.paths import ProjectPaths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full EfficientNet-B0 training pipeline")
    parser.add_argument("--crop", default="grape")
    return parser.parse_args()


def apply_training_config(config: AppConfig) -> AppConfig:
    """Apply required EfficientNet-B0 training settings."""
    config.raw.setdefault("training", {})
    training = config.raw["training"]
    training.update({
        "model_name": "efficientnet_b0",
        "num_epochs": 50,
        "batch_size": 32,
        "image_size": 224,
        "learning_rate": 0.001,
        "weight_decay": 1e-4,
        "mixed_precision": True,
        "gradient_clip_norm": 1.0,
        "use_class_weights": True,
        "tensorboard": True,
    })
    training.setdefault("early_stopping", {})
    training["early_stopping"].update({"enabled": True, "patience": 10})
    config.raw.setdefault("project", {})["seed"] = 42
    return config


def find_sample_test_image(test_dir: Path) -> Path:
    for class_dir in sorted(test_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                return path
    raise FileNotFoundError(f"No test images found in {test_dir}")


def verify_exported_models(
    config: AppConfig,
    sample_image: Path,
    torchscript_path: Path,
    onnx_path: Path,
) -> dict[str, str]:
    """Run one inference per exported backend."""
    results: dict[str, str] = {}

    predictor = Predictor(config, backend="pytorch")
    pytorch_result = predictor.predict(sample_image)
    results["pytorch"] = (
        f"{pytorch_result.predicted_class} ({pytorch_result.confidence:.2%})"
    )

    ts_predictor = Predictor(config, backend="torchscript")
    ts_result = ts_predictor.predict(sample_image)
    results["torchscript"] = f"{ts_result.predicted_class} ({ts_result.confidence:.2%})"

    onnx_predictor = Predictor(config, backend="onnx")
    onnx_result = onnx_predictor.predict(sample_image)
    results["onnx"] = f"{onnx_result.predicted_class} ({onnx_result.confidence:.2%})"

    return results


def main() -> int:
    args = parse_args()
    config = apply_training_config(load_config(crop=args.crop, project_root=PROJECT_ROOT))
    logger = setup_logging(config, log_name="pipeline", log_subdir="training")
    paths = ProjectPaths.from_config(config)

    logger.info("Starting EfficientNet-B0 training pipeline")
    device = get_device(config)
    logger.info("Device: %s", device)

    trainer = Trainer(config)
    train_result = trainer.train()
    if not train_result.success:
        for error in train_result.errors:
            logger.error(error)
        return 1

    best_model_path = paths.weights / "best_model.pth"
    last_model_path = paths.weights / "last_model.pth"

    logger.info("Evaluating on test set...")
    eval_result = Evaluator(config, weights_path=best_model_path).evaluate()
    if not eval_result.success or eval_result.metrics is None:
        for error in eval_result.errors:
            logger.error(error)
        return 1

    logger.info("Exporting production models...")
    export_result = ModelExporter(config, weights_path=best_model_path).export()
    if not export_result.success:
        for error in export_result.errors:
            logger.error(error)
        return 1

    prefix = str(config.get("export.model_name_prefix", "grape_disease"))
    export_dir = paths.exports
    torchscript_path = export_dir / f"{prefix}.torchscript.pt"
    onnx_path = export_dir / f"{prefix}.onnx"
    sample_image = find_sample_test_image(paths.test)

    logger.info("Verifying exports with sample image: %s", sample_image.name)
    verification = verify_exported_models(config, sample_image, torchscript_path, onnx_path)

    test_accuracy = eval_result.metrics.accuracy

    print()
    print("Training completed successfully.")
    print()
    print(f"Best Validation Accuracy: {train_result.best_val_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Best Model: {best_model_path.resolve()}")
    print(f"Last Model: {last_model_path.resolve()}")
    print(f"TorchScript: {torchscript_path.resolve()}")
    print(f"ONNX: {onnx_path.resolve()}")
    print(f"Metrics: {eval_result.metrics_path.resolve() if eval_result.metrics_path else 'N/A'}")
    print(f"Sample inference verification ({sample_image.name}):")
    for backend, prediction in verification.items():
        print(f"  {backend}: {prediction}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
