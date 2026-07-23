"""Tests for Grad-CAM explainability."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from inference.gradcam import (
    GradCAM,
    _normalize_cam,
    apply_colormap,
    create_overlay,
    get_efficientnet_b0_target_layer,
    save_gradcam_outputs,
)
from models.factory import ModelFactory

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestGradCAMUtils:
    def test_normalize_cam(self) -> None:
        cam = np.array([[0.0, 2.0], [4.0, 8.0]], dtype=np.float32)
        normalized = _normalize_cam(cam)
        assert normalized.min() == 0.0
        assert normalized.max() == 1.0

    def test_apply_colormap_shape(self) -> None:
        cam = np.random.rand(32, 32).astype(np.float32)
        colored = apply_colormap(cam)
        assert colored.shape == (32, 32, 3)
        assert colored.dtype == np.uint8

    def test_create_overlay_shape(self) -> None:
        original = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        cam = np.random.rand(32, 32).astype(np.float32)
        overlay = create_overlay(original, cam)
        assert overlay.shape == original.shape

    def test_efficientnet_target_layer(self) -> None:
        model = ModelFactory.create("efficientnet_b0", 4, pretrained=False)
        layer = get_efficientnet_b0_target_layer(model)
        assert isinstance(layer, nn.Conv2d)


class TestGradCAMGeneration:
    @pytest.fixture
    def model_and_input(self):
        model = ModelFactory.create("efficientnet_b0", 4, pretrained=False)
        model.eval()
        tensor = torch.randn(1, 3, 224, 224)
        return model, tensor

    def test_gradcam_produces_2d_map(self, model_and_input) -> None:
        model, tensor = model_and_input
        target_layer = get_efficientnet_b0_target_layer(model)
        with GradCAM(model, target_layer, torch.device("cpu")) as grad_cam:
            cam = grad_cam.generate(tensor, target_class=0)
        assert cam.ndim == 2
        assert cam.min() >= 0.0
        assert cam.max() <= 1.0

    def test_save_outputs(self, tmp_path: Path) -> None:
        original = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        cam = np.random.rand(32, 32).astype(np.float32)
        outputs = save_gradcam_outputs(original, cam, tmp_path)
        assert outputs.original_path.exists()
        assert outputs.heatmap_path.exists()
        assert outputs.overlay_path.exists()
        assert outputs.original_path.name == "original.jpg"
        assert outputs.heatmap_path.name == "heatmap.png"
        assert outputs.overlay_path.name == "overlay.png"


@pytest.mark.skipif(
    not (PROJECT_ROOT / "weights" / "grape" / "best_model.pth").exists(),
    reason="Trained weights not available",
)
class TestExplainablePredictorIntegration:
    def test_predict_with_gradcam_on_test_image(self, tmp_path: Path) -> None:
        from inference.explainable_predictor import ExplainablePredictor
        from utils.config import load_config

        config = load_config("grape", project_root=PROJECT_ROOT)
        test_dir = PROJECT_ROOT / "datasets" / "grape" / "test"
        sample = next(
            p for d in test_dir.iterdir() if d.is_dir()
            for p in d.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

        predictor = ExplainablePredictor(config, gradcam_output_root=tmp_path)
        baseline = predictor.predict(sample)
        result = predictor.predict_with_gradcam(sample, output_dir=tmp_path / sample.stem)

        assert result.predicted_class == baseline.predicted_class
        assert abs(result.confidence - baseline.confidence) < 1e-5
        assert Path(result.heatmap_path).exists()
        assert Path(result.overlay_path).exists()
        assert Path(result.original_output_path).exists()
        assert len(result.top_predictions) <= 3
