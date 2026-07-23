"""Tests for model factory."""

from __future__ import annotations

import torch

from models.factory import ModelFactory, SupportedModel, build_model


class TestModelFactory:
    def test_supported_models(self) -> None:
        models = ModelFactory.supported_models()
        assert SupportedModel.EFFICIENTNET_B0.value in models
        assert SupportedModel.MOBILENET_V3_LARGE.value in models

    def test_build_efficientnet(self) -> None:
        model = build_model("efficientnet_b0", num_classes=4, pretrained=False)
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        assert out.shape == (1, 4)

    def test_build_mobilenet(self) -> None:
        model = build_model("mobilenet_v3_large", num_classes=4, pretrained=False)
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        assert out.shape == (1, 4)

    def test_unsupported_model_raises(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="Unsupported model"):
            build_model("resnet50", num_classes=4)

    def test_factory_create_with_device(self) -> None:
        model = ModelFactory.create("efficientnet_b0", num_classes=4, pretrained=False, device=torch.device("cpu"))
        assert next(model.parameters()).device.type == "cpu"
