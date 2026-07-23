"""Tests for configuration loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from utils.config import AppConfig, load_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestConfig:
    def test_load_grape_config(self) -> None:
        config = load_config(crop="grape", project_root=PROJECT_ROOT)
        assert isinstance(config, AppConfig)
        assert config.crop_name == "grape"
        assert config.num_classes == 4

    def test_class_names(self) -> None:
        config = load_config(crop="grape", project_root=PROJECT_ROOT)
        names = config.class_names
        assert "Black Rot" in names
        assert "Healthy" in names
        assert len(names) == 4

    def test_nested_get(self) -> None:
        config = load_config(crop="grape", project_root=PROJECT_ROOT)
        assert config.get("training.batch_size") == 32
        assert config.get("nonexistent.key", "default") == "default"

    def test_class_folders(self) -> None:
        config = load_config(crop="grape", project_root=PROJECT_ROOT)
        folders = config.class_folders
        assert "Black_Rot" in folders
        assert "Healthy" in folders

    def test_path_resolution(self) -> None:
        config = load_config(crop="grape", project_root=PROJECT_ROOT)
        raw_path = config.path("paths.raw")
        assert raw_path == (PROJECT_ROOT / "datasets/grape/raw").resolve()
