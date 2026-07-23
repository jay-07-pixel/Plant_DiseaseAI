"""Webcam / Pi Camera capture dialog."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import cv2
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from desktop_app.i18n import Translator
from desktop_app.services.camera_service import CameraBackend, create_camera_backend
from utils.config import AppConfig


class CameraCaptureDialog(QDialog):
    """Capture a single frame from USB webcam or Raspberry Pi Camera Module."""

    def __init__(
        self,
        config: AppConfig,
        translator: Translator,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.config = config
        self.translator = translator
        self.captured_path: Path | None = None
        self._camera: CameraBackend | None = None
        self._timer = QTimer(self)

        self.setFixedSize(640, 520)

        layout = QVBoxLayout(self)
        self.preview = QLabel()
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(600, 400)
        self.preview.setStyleSheet("background: #F0F2F5; border-radius: 8px;")

        btn_row = QHBoxLayout()
        self.capture_btn = QPushButton()
        self.cancel_btn = QPushButton()
        self.cancel_btn.setObjectName("secondaryBtn")
        btn_row.addStretch()
        btn_row.addWidget(self.capture_btn)
        btn_row.addWidget(self.cancel_btn)

        layout.addWidget(self.preview)
        layout.addLayout(btn_row)

        self.capture_btn.clicked.connect(self._capture)
        self.cancel_btn.clicked.connect(self.reject)
        self._timer.timeout.connect(self._update_frame)

        self.retranslate_ui()

    def retranslate_ui(self) -> None:
        self.setWindowTitle(self.translator.t("camera.title"))
        self.capture_btn.setText(self.translator.t("buttons.capture"))
        self.cancel_btn.setText(self.translator.t("buttons.cancel"))
        if self._camera is None or not self._camera.is_open:
            if not self.preview.pixmap() or self.preview.pixmap().isNull():
                self.preview.setText(self.translator.t("camera.initializing"))

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._camera = create_camera_backend(self.config)
        if self._camera is None or not self._camera.is_open:
            self.preview.setText(self.translator.t("camera.not_available"))
            self.capture_btn.setEnabled(False)
            return

        self.preview.setText(self.translator.t("camera.initializing"))
        self._timer.start(30)

    def closeEvent(self, event) -> None:
        self._timer.stop()
        if self._camera is not None:
            self._camera.close()
            self._camera = None
        super().closeEvent(event)

    def _frame_to_pixmap(self, rgb_frame) -> QPixmap:
        h, w, ch = rgb_frame.shape
        image = QImage(rgb_frame.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
        return QPixmap.fromImage(image).scaled(
            self.preview.width(),
            self.preview.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    def _update_frame(self) -> None:
        if self._camera is None:
            return
        frame = self._camera.read_rgb()
        if frame is None:
            return
        self.preview.setPixmap(self._frame_to_pixmap(frame))

    def _capture(self) -> None:
        if self._camera is None:
            return
        frame = self._camera.read_rgb()
        if frame is None:
            return

        capture_dir = self.config.project_root / "logs" / "captures"
        capture_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = capture_dir / f"capture_{timestamp}.jpg"
        cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        self.captured_path = path
        self.accept()
