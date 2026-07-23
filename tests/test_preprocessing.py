"""Tests for label validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from preprocessing.validators import validate_labels_and_folders
from utils.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestValidators:
    @pytest.fixture
    def config(self):
        return load_config(crop="grape", project_root=PROJECT_ROOT)

    def test_missing_raw_dir(self, config, tmp_path: Path) -> None:
        result = validate_labels_and_folders(tmp_path / "nonexistent", config)
        assert not result.is_valid
        assert len(result.errors) > 0

    def test_valid_structure(self, config, tmp_path: Path) -> None:
        raw = tmp_path / "raw"
        for folder in config.class_folders:
            (raw / folder).mkdir(parents=True)
            (raw / folder / "test.jpg").write_bytes(b"\xff\xd8\xff")

        result = validate_labels_and_folders(raw, config)
        assert result.is_valid
        assert not result.missing_folders

    def test_missing_class_folder(self, config, tmp_path: Path) -> None:
        raw = tmp_path / "raw"
        (raw / "Healthy").mkdir(parents=True)

        result = validate_labels_and_folders(raw, config)
        assert not result.is_valid
        assert len(result.missing_folders) == 3

    def test_plantvillage_aliases(self, config, tmp_path: Path) -> None:
        raw = tmp_path / "raw"
        alias_folders = [
            "Grape___Black_rot",
            "Grape___Esca_(Black_Measles)",
            "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
            "Grape___healthy",
        ]
        for folder in alias_folders:
            (raw / folder).mkdir(parents=True)
            (raw / folder / "img.jpg").write_bytes(b"\xff\xd8\xff")

        result = validate_labels_and_folders(raw, config)
        assert result.is_valid
        assert len(result.alias_mappings) == 4
