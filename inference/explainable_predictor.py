"""Explainable inference with integrated Grad-CAM."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from inference.gradcam import GradCAMOutputs, generate_gradcam
from inference.predictor import Predictor, PredictionResult, TopPrediction
from utils.image_utils import read_image_rgb

logger = logging.getLogger(__name__)


@dataclass
class ExplainablePredictionResult(PredictionResult):
    """Prediction result extended with Grad-CAM explainability artifacts."""

    heatmap_path: str | None = None
    overlay_path: str | None = None
    original_output_path: str | None = None
    gradcam_output_dir: str | None = None
    heatmap: np.ndarray | None = field(default=None, repr=False)

    def to_dict(self) -> dict:
        base = super().to_dict()
        base.update({
            "heatmap_path": self.heatmap_path,
            "overlay_path": self.overlay_path,
            "original_output_path": self.original_output_path,
            "gradcam_output_dir": self.gradcam_output_dir,
        })
        return base


class ExplainablePredictor(Predictor):
    """
    PyTorch inference predictor with automatic Grad-CAM generation.

    Loads crop-specific weights and class mapping from configuration.
    Every call to ``predict_with_gradcam`` runs standard prediction first,
    then generates a Top-1 Grad-CAM overlay without altering predictions.
    """

    def __init__(
        self,
        config,
        weights_path: Path | None = None,
        gradcam_output_root: Path | None = None,
        overlay_alpha: float = 0.45,
    ) -> None:
        super().__init__(
            config,
            weights_path=weights_path,
            backend="pytorch",
        )
        self.gradcam_output_root = gradcam_output_root or (
            config.project_root / "outputs" / "gradcam" / config.crop_name
        )
        self.overlay_alpha = overlay_alpha

    def _gradcam_enabled(self) -> bool:
        env = os.getenv("PLANT_DISEASE_ENABLE_GRADCAM", "").strip().lower()
        if env in {"0", "false", "no", "off"}:
            return False
        if env in {"1", "true", "yes", "on"}:
            return True
        return bool(self.config.get("inference.enable_gradcam", True))

    def _resolve_output_dir(self, image_path: Path, output_dir: Path | None) -> Path:
        if output_dir is not None:
            return output_dir
        return self.gradcam_output_root / image_path.stem

    def predict_with_gradcam(
        self,
        image_path: Path | str,
        output_dir: Path | str | None = None,
    ) -> ExplainablePredictionResult:
        """
        Run prediction and generate Grad-CAM for the Top-1 predicted class.

        Saves ``original.jpg``, ``heatmap.png``, and ``overlay.png`` under
        ``outputs/gradcam/<image_stem>/`` by default.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        # Step 1: Standard prediction (no gradients, unchanged logic)
        base_result = self.predict(path)

        if not self._gradcam_enabled():
            logger.info("Grad-CAM disabled; returning prediction only")
            original_rgb = read_image_rgb(path)
            out_dir = self._resolve_output_dir(path, Path(output_dir) if output_dir else None)
            out_dir.mkdir(parents=True, exist_ok=True)
            original_path = out_dir / "original.jpg"
            try:
                import cv2

                cv2.imwrite(str(original_path), cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR))
            except Exception:
                original_path = path

            return ExplainablePredictionResult(
                predicted_class=base_result.predicted_class,
                predicted_class_id=base_result.predicted_class_id,
                confidence=base_result.confidence,
                top_predictions=list(base_result.top_predictions),
                inference_time_ms=base_result.inference_time_ms,
                image_path=str(path),
                heatmap_path=None,
                overlay_path=None,
                original_output_path=str(original_path),
                gradcam_output_dir=str(out_dir),
                heatmap=None,
            )

        if self._model is None or not isinstance(self._model, torch.nn.Module):
            raise RuntimeError("Grad-CAM requires the PyTorch backend and a loaded nn.Module.")

        # Step 2: Grad-CAM for Top-1 class
        original_rgb = read_image_rgb(path)
        tensor = self._preprocess(path)
        out_dir = self._resolve_output_dir(path, Path(output_dir) if output_dir else None)

        gradcam_start = time.perf_counter()
        gradcam_outputs: GradCAMOutputs = generate_gradcam(
            model=self._model,
            input_tensor=tensor,
            target_class=base_result.predicted_class_id,
            original_rgb=original_rgb,
            output_dir=out_dir,
            device=self.device,
            alpha=self.overlay_alpha,
        )
        gradcam_ms = (time.perf_counter() - gradcam_start) * 1000

        return ExplainablePredictionResult(
            predicted_class=base_result.predicted_class,
            predicted_class_id=base_result.predicted_class_id,
            confidence=base_result.confidence,
            top_predictions=list(base_result.top_predictions),
            inference_time_ms=base_result.inference_time_ms + gradcam_ms,
            image_path=str(path),
            heatmap_path=str(gradcam_outputs.heatmap_path),
            overlay_path=str(gradcam_outputs.overlay_path),
            original_output_path=str(gradcam_outputs.original_path),
            gradcam_output_dir=str(out_dir),
            heatmap=gradcam_outputs.heatmap,
        )

    def predict(self, image_path: Path | str) -> PredictionResult:
        """Standard prediction without Grad-CAM (backward compatible)."""
        return super().predict(image_path)
