"""Tests for disease information database."""

from __future__ import annotations

from desktop_app.disease_info import DISEASE_DATABASE, get_disease_info


class TestDiseaseInfo:
    def test_all_classes_have_info(self) -> None:
        expected = [
            "Black Rot",
            "Esca (Black Measles)",
            "Leaf Blight (Isariopsis Leaf Spot)",
            "Healthy",
        ]
        for name in expected:
            assert name in DISEASE_DATABASE

    def test_get_disease_info(self) -> None:
        info = get_disease_info("Black Rot")
        assert info.name == "Black Rot"
        assert len(info.symptoms) > 0
        assert len(info.treatment) > 0
        assert len(info.prevention) > 0

    def test_healthy_severity(self) -> None:
        info = get_disease_info("Healthy")
        assert info.severity == "low"

    def test_unknown_class_fallback(self) -> None:
        info = get_disease_info("Unknown Disease")
        assert info.name == "Unknown Disease"
