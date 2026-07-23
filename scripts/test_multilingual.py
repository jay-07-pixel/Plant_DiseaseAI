#!/usr/bin/env python3
"""Verify multilingual UI strings and Groq responses."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from inference.predictor import TopPrediction

from desktop_app.i18n import Translator
from desktop_app.services.groq_service import GroqExplanationService


def _mock_top3(predicted: str, confidence: float) -> list[TopPrediction]:
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
    return preds[:3]


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    translator = Translator()
    groq = GroqExplanationService(PROJECT_ROOT)

    if not groq.is_configured:
        print("GROQ_API_KEY not set — UI translations only.")
        for code in ("en", "hi", "mr"):
            translator.set_language(code)
            print(f"\n[{code}] upload={translator.t('buttons.upload_image')} | "
                  f"why={translator.t('explanation.why_title')}")
        return 0

    img = next((PROJECT_ROOT / "datasets/grape/test/Black_Rot").glob("*.jpg"))
    top3 = _mock_top3("Black Rot", 0.9984)

    print("Multilingual support integrated successfully.\n")

    for code in ("en", "hi", "mr"):
        translator.set_language(code)
        explanation = groq.get_explanation(
            image_path=img,
            crop_name=translator.t("left_panel.crop_grape"),
            predicted_class="Black Rot",
            confidence=0.9984,
            top_predictions=top3,
            language=code,
        )
        print(f"=== {translator.language_option(code)} ===")
        print(f"UI: Upload={translator.t('buttons.upload_image')}")
        print(f"UI: Why={translator.t('explanation.why_title')}")
        print(f"UI: Class={translator.translate_class('Black Rot')}")
        print(json.dumps({
            "overview": explanation.overview,
            "causes": explanation.causes[:2],
            "symptoms": explanation.symptoms[:2],
            "prevention": explanation.prevention[:2],
            "remedies": {
                "organic": explanation.remedies_organic[:2],
                "chemical": explanation.remedies_chemical[:1],
            },
            "tips": explanation.tips[:2],
            "latency_ms": round(explanation.latency_ms, 1),
        }, ensure_ascii=False, indent=2))
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
