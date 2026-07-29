"""Application controller — connects UI events to services."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from PySide6.QtWidgets import QFileDialog, QMessageBox, QWidget

from desktop_app.services.groq_service import GroqExplanation, GroqExplanationService
from desktop_app.services.inference_service import InferenceService
from desktop_app.services.model_manager import ModelManager
from desktop_app.services.workers import GroqWorker, InferenceWorker, ModelLoadWorker
from desktop_app.widgets.camera_dialog import CameraCaptureDialog
from inference.explainable_predictor import ExplainablePredictionResult
from utils.platform import prepare_image_for_pi_inference
from utils.config import AppConfig

logger = logging.getLogger(__name__)


class AppController:
    """
    Mediates between the main window UI and the inference service.

    Keeps inference logic out of widget code.
    """

    def __init__(self, config: AppConfig, window: QWidget) -> None:
        self.config = config
        self.window = window
        self.model_manager = ModelManager(config.project_root)
        self.groq_service = GroqExplanationService(config.project_root)
        self.service: InferenceService | None = None
        self._current_crop = config.crop_name
        self._inference_worker: InferenceWorker | None = None
        self._groq_worker: GroqWorker | None = None
        self._load_worker: ModelLoadWorker | None = None
        self._current_image_path: Path | None = None
        self._last_result: ExplainablePredictionResult | None = None
        self._inference_generation = 0

        self._bind_window()

    def _bind_window(self) -> None:
        w = self.window
        w.upload_btn.clicked.connect(self.on_upload)
        w.capture_btn.clicked.connect(self.on_capture)
        w.left_panel.crop_combo.currentIndexChanged.connect(self.on_crop_changed)

    def _t(self, key: str, **kwargs) -> str:
        return self.window.translator.t(key, **kwargs)

    def start_model_loading(self) -> None:
        self._load_crop_model(self._current_crop)

    def on_crop_changed(self, _index: int = 0) -> None:
        crop = self.window.left_panel.selected_crop()
        if crop == self._current_crop and self.service is not None:
            return
        self._inference_generation += 1
        self._current_crop = crop
        self._last_result = None
        self._current_image_path = None
        self.window.clear_inference_state()
        self._load_crop_model(crop)

    def _load_crop_model(self, crop: str) -> None:
        if self.model_manager.is_cached(crop):
            self.config = self.model_manager.load_config(crop)
            self.service = self.model_manager.get_cached(crop)
            self.window.set_model_ready(self.service)
            logger.info("Model loaded from cache | crop=%s", crop)
            return

        self.window.set_model_loading()
        self._load_worker = ModelLoadWorker(crop, self.model_manager, self.window)
        self._load_worker.loaded.connect(self._on_model_loaded)
        self._load_worker.error.connect(self._on_model_load_error)
        self._load_worker.start()

    def _on_model_loaded(self, crop: str, service: InferenceService) -> None:
        self._current_crop = crop
        self.config = self.model_manager.load_config(crop)
        self.service = service
        self.window.set_model_ready(service)
        logger.info("Model loaded for desktop app | crop=%s", crop)

    def _on_model_load_error(self, error: str) -> None:
        self.window.set_model_error(error)
        logger.error("Model load failed: %s", error)
        QMessageBox.critical(
            self.window,
            self._t("dialog.model_load_error_title"),
            self._t("dialog.model_load_error_body", error=error),
        )

    def on_upload(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self.window,
            self._t("file_dialog.select_leaf"),
            "",
            self._t("file_dialog.image_filter"),
        )
        if path:
            self.run_inference(Path(path))

    def on_capture(self) -> None:
        dialog = CameraCaptureDialog(self.config, self.window.translator, self.window)
        if dialog.exec() and dialog.captured_path:
            # Capture file is under project_root/logs/captures (pendrive when app
            # is launched from the USB project folder).
            self.run_inference(dialog.captured_path)

    def run_inference(self, image_path: Path) -> None:
        selected_crop = self.window.left_panel.selected_crop()

        if self.service is None or not self.service.is_loaded:
            QMessageBox.warning(
                self.window,
                self._t("dialog.model_not_ready_title"),
                self._t("dialog.model_not_ready_body"),
            )
            return

        if self.service.crop_name != selected_crop:
            QMessageBox.warning(
                self.window,
                self._t("dialog.crop_mismatch_title"),
                self._t(
                    "dialog.crop_mismatch_body",
                    selected=self.window.left_panel.selected_crop_display(),
                    loaded=self.service.crop_name.capitalize(),
                ),
            )
            return

        self._inference_generation += 1
        inference_generation = self._inference_generation
        prepared = prepare_image_for_pi_inference(
            image_path,
            output_dir=self.config.project_root / "logs" / "captures",
        )
        self._current_image_path = prepared
        if os.getenv("PLANT_DISEASE_INFERENCE_DEBUG", "").lower() in ("1", "true", "yes"):
            logger.info(
                "[INFERENCE_DEBUG] selected_crop=%s display=%s service_crop=%s weights=%s",
                selected_crop,
                self.window.left_panel.selected_crop_display(),
                self.service.crop_name,
                self.service.weights_path,
            )
        self.window.set_inference_running(prepared)
        self._inference_worker = InferenceWorker(
            self.service,
            prepared,
            inference_generation,
            self.window,
        )
        self._inference_worker.finished.connect(self._on_inference_finished)
        self._inference_worker.error.connect(self._on_inference_error)
        self._inference_worker.start()

    def _on_inference_finished(self, generation: int, result: ExplainablePredictionResult) -> None:
        if generation != self._inference_generation:
            logger.info("Discarding stale inference result (crop changed during prediction)")
            return
        if self.service and self.service.crop_name != self.window.left_panel.selected_crop():
            logger.warning("Discarding inference result due to crop mismatch")
            return

        self._last_result = result
        low_confidence = result.confidence < 0.75
        if len(result.top_predictions) >= 2:
            margin = result.top_predictions[0].confidence - result.top_predictions[1].confidence
            low_confidence = low_confidence or margin < 0.15

        self.window.display_result(result, low_confidence=low_confidence)
        logger.info(
            "Inference complete | crop=%s prediction=%s confidence=%.2f",
            self._current_crop,
            result.predicted_class,
            result.confidence,
        )
        self._request_groq_explanation(result)

    def _request_groq_explanation(self, result: ExplainablePredictionResult) -> None:
        if self._current_image_path is None:
            return

        self.window.set_groq_loading()
        crop_display = self.window.left_panel.selected_crop_display()
        language = self.window.translator.locale

        self._groq_worker = GroqWorker(
            self.groq_service,
            self._current_image_path,
            crop_display,
            self._current_crop,
            result,
            language,
            self.window,
        )
        self._groq_worker.finished.connect(self._on_groq_finished)
        self._groq_worker.error.connect(self._on_groq_error)
        self._groq_worker.start()

    def on_language_changed(self) -> None:
        if self._last_result is None or self._current_image_path is None:
            return
        self._request_groq_explanation(self._last_result)

    def _on_groq_finished(self, explanation: GroqExplanation) -> None:
        self.window.display_groq_explanation(explanation)
        if not explanation.from_cache:
            logger.info(
                "Groq explanation displayed | crop=%s language=%s latency_ms=%.0f",
                self._current_crop,
                self.window.translator.locale,
                explanation.latency_ms,
            )

    def _on_groq_error(self, error: str) -> None:
        logger.error("Groq explanation failed: %s", error)
        self.window.display_groq_unavailable()

    def _on_inference_error(self, error: str) -> None:
        self.window.set_inference_error(error)
        logger.error("Inference error: %s", error)
        QMessageBox.critical(
            self.window,
            self._t("dialog.inference_error_title"),
            self._t("dialog.inference_error_body", error=error),
        )
