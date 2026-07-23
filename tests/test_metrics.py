"""Tests for evaluation metrics."""

from __future__ import annotations

import numpy as np

from evaluation.metrics import compute_metrics


class TestMetrics:
    def test_perfect_predictions(self) -> None:
        y_true = np.array([0, 1, 2, 3, 0, 1])
        y_pred = np.array([0, 1, 2, 3, 0, 1])
        y_prob = np.eye(4)[y_pred]

        class_names = ["A", "B", "C", "D"]
        result = compute_metrics(y_true, y_pred, y_prob, class_names)

        assert result.accuracy == 1.0
        assert result.f1_macro == 1.0

    def test_confusion_matrix_shape(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        y_prob = np.random.rand(4, 2)
        y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)

        result = compute_metrics(y_true, y_pred, y_prob, ["A", "B"])
        assert len(result.confusion_matrix) == 2
        assert len(result.confusion_matrix[0]) == 2

    def test_to_dict(self) -> None:
        y_true = np.array([0, 1])
        y_pred = np.array([0, 1])
        result = compute_metrics(y_true, y_pred, None, ["A", "B"])
        d = result.to_dict()
        assert "accuracy" in d
        assert "confusion_matrix" in d
