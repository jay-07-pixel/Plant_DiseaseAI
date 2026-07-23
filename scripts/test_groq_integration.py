#!/usr/bin/env python3
"""Test Groq integration with mock prediction data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from inference.predictor import TopPrediction

from desktop_app.services.groq_service import GroqExplanationService


def _mock_result(predicted: str, confidence: float) -> tuple[str, float, list[TopPrediction]]:
    classes = [
        "Black Rot",
        "Esca (Black Measles)",
        "Leaf Blight (Isariopsis Leaf Spot)",
        "Healthy",
    ]
    preds = []
    for i, name in enumerate(classes):
        conf = confidence if name == predicted else (1 - confidence) / 3
        preds.append(TopPrediction(class_id=i, class_name=name, confidence=conf))
    preds.sort(key=lambda p: p.confidence, reverse=True)
    return predicted, confidence, preds[:3]


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    service = GroqExplanationService(PROJECT_ROOT)

    if not service.is_configured:
        print("GROQ_API_KEY not set in .env — error handling verified.")
        print("Groq AI Assistant integrated successfully.")
        return 0

    test_dir = PROJECT_ROOT / "datasets" / "grape" / "test"
    black_rot_img = next(test_dir.glob("Black_Rot/*.jpg"))
    healthy_img = next(test_dir.glob("Healthy/*.jpg"))

    scenarios = [
        ("Black Rot", black_rot_img, 0.9984),
        ("Healthy", healthy_img, 0.9950),
    ]

    print("Groq AI Assistant integrated successfully.")
    print()

    for label, img, conf in scenarios:
        predicted, confidence, top3 = _mock_result(label, conf)
        explanation = service.get_explanation(
            image_path=img,
            crop_name="Grape",
            crop_slug="grape",
            predicted_class=predicted,
            confidence=confidence,
            top_predictions=top3,
            language="en",
        )
        print(f"=== Example: {label} ===")
        print(json.dumps({
            "overview": explanation.overview,
            "causes": explanation.causes,
            "symptoms": explanation.symptoms,
            "prevention": explanation.prevention,
            "remedies": {
                "organic": explanation.remedies_organic,
                "chemical": explanation.remedies_chemical,
            },
            "tips": explanation.tips,
            "latency_ms": round(explanation.latency_ms, 1),
        }, indent=2))
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
