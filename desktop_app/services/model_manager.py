"""Cached multi-crop model loading for the desktop application."""

from __future__ import annotations

from pathlib import Path

from desktop_app.services.inference_service import InferenceService
from utils.config import AppConfig, apply_platform_overrides, load_config


class ModelManager:
    """Load and cache inference services per crop without reloading duplicates."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self._cache: dict[str, InferenceService] = {}

    def is_cached(self, crop: str) -> bool:
        return crop in self._cache

    def get_cached(self, crop: str) -> InferenceService | None:
        return self._cache.get(crop)

    def load_config(self, crop: str) -> AppConfig:
        return apply_platform_overrides(load_config(crop=crop, project_root=self.project_root))

    def get_or_load(self, crop: str) -> InferenceService:
        cached = self._cache.get(crop)
        if cached is not None:
            return cached

        config = self.load_config(crop)
        service = InferenceService(config)
        service.load_model()
        self._cache[crop] = service
        return service
