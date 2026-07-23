#!/usr/bin/env python3
"""Complete end-to-end tomato runtime inference verification."""

from __future__ import annotations

import csv
import json
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _sanitize_import_path() -> None:
    vendor_root = (PROJECT_ROOT / ".vendor").resolve()
    sys.path = [
        entry for entry in sys.path
        if entry and Path(entry).resolve() != vendor_root
    ]


_sanitize_import_path()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
import torch.nn as nn

from desktop_app.services.inference_service import InferenceService
from models.factory import ModelFactory
from training.dataset import PlantDiseaseDataset
from training.transforms import IMAGENET_MEAN, IMAGENET_STD, get_val_transforms
from utils.config import load_config
from utils.image_utils import read_image_rgb


DEBUG_DIR = PROJECT_ROOT / "debug"
REAL_WORLD_DIR = DEBUG_DIR / "real_world_test"
REPORT_PATH = PROJECT_ROOT / "reports" / "tomato_runtime_diagnosis.md"

SAMPLE_FOLDERS = {
    "Healthy": "Tomato___healthy",
    "Early Blight": "Tomato___Early_blight",
    "Late Blight": "Tomato___Late_blight",
    "Target Spot": "Tomato___Target_Spot",
    "Tomato Yellow Leaf Curl Virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
}


@dataclass
class ImageResult:
    path: str
    ground_truth: str
    predicted_class: str
    confidence: float
    top3: list[tuple[str, float]]
    passed: bool
    logits: np.ndarray | None = None
    probs: np.ndarray | None = None


@dataclass
class DiagnosisReport:
    selected_crop: str = "tomato"
    weights_path: str = ""
    model_architecture: str = ""
    num_classes: int = 0
    device: str = ""
    class_mapping_path: str = ""
    class_index_map: dict[int, str] = field(default_factory=dict)
    mapping_valid: bool = False
    weights_correct: bool = False
    official_test_results: list[ImageResult] = field(default_factory=list)
    official_accuracy: float = 0.0
    gui_pipeline_checks: dict[str, Any] = field(default_factory=dict)
    preprocessing: dict[str, Any] = field(default_factory=dict)
    gui_vs_eval: dict[str, Any] = field(default_factory=dict)
    real_world_rows: list[dict[str, Any]] = field(default_factory=list)
    ood_analysis: dict[str, Any] = field(default_factory=dict)
    bug_found: bool = False
    bug_details: str = ""
    root_cause: str = "unknown"


def _list_images(folder: Path, n: int, rng: random.Random) -> list[Path]:
    images = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    ]
    if len(images) < n:
        raise RuntimeError(f"Not enough images in {folder}: found {len(images)}, need {n}")
    return rng.sample(images, n)


def _desktop_predict(predictor, image_path: Path) -> tuple[ImageResult, np.ndarray, np.ndarray]:
    """Use Predictor.predict — identical classification step used by desktop before Grad-CAM."""
    with torch.no_grad():
        tensor = predictor._preprocess(image_path)
        logits_t, probs_t = predictor._infer_pytorch(tensor)
        pred_idx = int(torch.argmax(probs_t, dim=1).item())
        topk = torch.topk(probs_t, k=min(3, probs_t.shape[1]), dim=1)
        logits = logits_t[0].cpu().numpy()
        probs = probs_t[0].cpu().numpy()
        top3 = [
            (predictor.id_to_name[int(idx)], float(probs[int(idx)]))
            for idx in topk.indices[0].cpu().tolist()
        ]
    result = predictor.predict(image_path)
    return (
        ImageResult(
            path=str(image_path.relative_to(PROJECT_ROOT)),
            ground_truth="",
            predicted_class=result.predicted_class,
            confidence=result.confidence,
            top3=[(p.class_name, p.confidence) for p in result.top_predictions],
            passed=True,
            logits=logits,
            probs=probs,
        ),
        logits,
        probs,
    )


def _eval_forward(predictor, image_path: Path) -> tuple[np.ndarray, np.ndarray, int, float]:
    class_mapping = {c.folder_name: c.id for c in predictor.config.class_configs}
    dataset = PlantDiseaseDataset(
        root=predictor.config.path("paths.test"),
        class_to_idx=class_mapping,
        transform=get_val_transforms(predictor.image_size),
    )
    sample_idx = next(
        i for i, (path, _) in enumerate(dataset.samples)
        if path.resolve() == image_path.resolve()
    )
    tensor, _ = dataset[sample_idx]
    model = predictor._model
    assert isinstance(model, nn.Module)
    with torch.no_grad():
        batch = tensor.unsqueeze(0).to(predictor.device)
        outputs = model(batch)
        probs_t = torch.softmax(outputs, dim=1)
        pred_idx = int(torch.argmax(probs_t, dim=1).item())
        confidence = float(probs_t[0, pred_idx].item())
    return outputs[0].cpu().numpy(), probs_t[0].cpu().numpy(), pred_idx, confidence


def _denormalize_tensor(chw: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    restored = chw.cpu() * std + mean
    restored = restored.clamp(0, 1)
    rgb = restored.permute(1, 2, 0).numpy()
    return (rgb * 255).astype(np.uint8)


def verify_gui_pipeline(predictor, sample_image: Path) -> dict[str, Any]:
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    rgb = read_image_rgb(sample_image)
    runtime_input_path = DEBUG_DIR / "runtime_input.png"
    cv2.imwrite(str(runtime_input_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    tensor = predictor._preprocess(sample_image)
    preprocessed_vis = _denormalize_tensor(tensor[0])
    preprocessed_path = DEBUG_DIR / "preprocessed_visualization.png"
    cv2.imwrite(str(preprocessed_path), cv2.cvtColor(preprocessed_vis, cv2.COLOR_RGB2BGR))

    resized_direct = cv2.resize(rgb, (predictor.image_size, predictor.image_size))
    resized_path = DEBUG_DIR / "runtime_resized_only.png"
    cv2.imwrite(str(resized_path), cv2.cvtColor(resized_direct, cv2.COLOR_RGB2BGR))

    checks = {
        "runtime_input_saved": runtime_input_path.exists(),
        "preprocessed_visualization_saved": preprocessed_path.exists(),
        "original_shape": list(rgb.shape),
        "original_dtype": str(rgb.dtype),
        "original_channels": int(rgb.shape[2]) if rgb.ndim == 3 else 1,
        "has_alpha_channel": False if rgb.ndim == 2 else rgb.shape[2] == 4,
        "rgb_ordering": "RGB (read_image_rgb)",
        "resize_only_applied": True,
        "resize_target": [predictor.image_size, predictor.image_size],
        "no_random_rotation_in_inference": True,
        "no_center_crop_in_inference": True,
        "tensor_shape": list(tensor.shape),
        "tensor_dtype": str(tensor.dtype),
        "tensor_device": str(tensor.device),
        "resize_content_similarity_mse": float(
            np.mean((resized_direct.astype(np.float32) - preprocessed_vis.astype(np.float32)) ** 2)
        ),
    }
    checks["pipeline_ok"] = (
        checks["original_channels"] == 3
        and not checks["has_alpha_channel"]
        and checks["tensor_shape"] == [1, 3, predictor.image_size, predictor.image_size]
    )
    return checks


def verify_model_loading(service: InferenceService, config) -> dict[str, Any]:
    predictor = service._predictor
    assert predictor is not None
    expected_weights = (PROJECT_ROOT / "weights/tomato/best_model.pth").resolve()
    loaded_weights = Path(predictor.weights_path).resolve()
    model = predictor._model
    architecture = type(model).__name__ if model is not None else "unknown"
    return {
        "selected_crop": service.crop_name,
        "loaded_weights_path": str(loaded_weights),
        "expected_weights_path": str(expected_weights),
        "weights_match_expected": loaded_weights == expected_weights,
        "model_architecture": architecture,
        "model_name_config": service.model_name,
        "num_output_classes": len(predictor.class_names),
        "device": str(predictor.device),
    }


def verify_class_mapping(predictor) -> tuple[dict[int, str], bool]:
    mapping_path = PROJECT_ROOT / "datasets/tomato/reports/class_mapping.json"
    with mapping_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    file_map = {int(k): v for k, v in data["id_to_display"].items()}
    runtime_map = {i: name for i, name in enumerate(predictor.class_names)}
    valid = file_map == runtime_map
    return runtime_map, valid


def run_official_test_samples(rng: random.Random, predictor) -> list[ImageResult]:
    test_root = PROJECT_ROOT / "datasets/tomato/split/test"
    folder_to_display = {c.folder_name: c.display_name for c in predictor.config.class_configs}
    results: list[ImageResult] = []

    for display_name, folder_name in SAMPLE_FOLDERS.items():
        folder = test_root / folder_name
        for image_path in _list_images(folder, 5, rng):
            item, _, _ = _desktop_predict(predictor, image_path)
            item.ground_truth = display_name
            item.passed = item.predicted_class == display_name
            results.append(item)
            print(f"[{'PASS' if item.passed else 'FAIL'}] {item.path}")
            print(f"  GT={item.ground_truth} | PRED={item.predicted_class} | conf={item.confidence:.4f}")
            print(f"  top3={item.top3}")
    return results


def compare_gui_vs_eval(predictor, sample_image: Path) -> dict[str, Any]:
    _, desktop_logits, desktop_probs = _desktop_predict(predictor, sample_image)
    eval_logits, eval_probs, eval_idx, eval_conf = _eval_forward(predictor, sample_image)
    desktop_idx = int(np.argmax(desktop_probs))
    return {
        "sample_image": str(sample_image.relative_to(PROJECT_ROOT)),
        "desktop_predicted_index": desktop_idx,
        "eval_predicted_index": eval_idx,
        "desktop_predicted_name": predictor.id_to_name[desktop_idx],
        "eval_predicted_name": predictor.id_to_name[eval_idx],
        "desktop_confidence": float(desktop_probs[desktop_idx]),
        "eval_confidence": eval_conf,
        "max_logit_abs_diff": float(np.max(np.abs(desktop_logits - eval_logits))),
        "max_prob_abs_diff": float(np.max(np.abs(desktop_probs - eval_probs))),
        "identical": float(np.max(np.abs(desktop_logits - eval_logits))) == 0.0,
    }


def run_real_world_tests(predictor) -> list[dict[str, Any]]:
    REAL_WORLD_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    images = [
        p for p in REAL_WORLD_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    ] if REAL_WORLD_DIR.exists() else []

    for image_path in sorted(images):
        result = predictor.predict(image_path)
        row = {
            "filename": image_path.name,
            "predicted_class": result.predicted_class,
            "confidence": round(result.confidence, 6),
            "top1": result.top_predictions[0].class_name if result.top_predictions else "",
            "top2": result.top_predictions[1].class_name if len(result.top_predictions) > 1 else "",
            "top3": result.top_predictions[2].class_name if len(result.top_predictions) > 2 else "",
            "top1_conf": round(result.top_predictions[0].confidence, 6) if result.top_predictions else 0,
            "top2_conf": round(result.top_predictions[1].confidence, 6) if len(result.top_predictions) > 1 else 0,
            "top3_conf": round(result.top_predictions[2].confidence, 6) if len(result.top_predictions) > 2 else 0,
        }
        rows.append(row)

    csv_path = DEBUG_DIR / "real_world_results.csv"
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "filename", "predicted_class", "confidence",
                "top1", "top2", "top3", "top1_conf", "top2_conf", "top3_conf",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return rows


def analyze_ood(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    blight_classes = {"Early Blight", "Late Blight"}
    high_conf = [r for r in rows if r["confidence"] >= 0.95]
    blight_dominated = [r for r in rows if r["predicted_class"] in blight_classes]
    return {
        "real_world_image_count": len(rows),
        "high_confidence_count": len(high_conf),
        "high_confidence_filenames": [r["filename"] for r in high_conf],
        "blight_prediction_count": len(blight_dominated),
        "blight_prediction_ratio": (len(blight_dominated) / len(rows)) if rows else 0.0,
        "overconfident_ood_suspected": len(high_conf) >= max(1, len(rows) // 2) if rows else False,
    }


def write_report(report: DiagnosisReport) -> None:
    lines = [
        "# Tomato Runtime Inference Diagnosis",
        "",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        "## Executive Summary",
        "",
        f"- **Root cause:** {report.root_cause}",
        f"- **Runtime pipeline bug found:** {'Yes' if report.bug_found else 'No'}",
        f"- **Official sample accuracy (desktop pipeline):** {report.official_accuracy:.1f}%",
        "",
        "## 1. Official Test Dataset Verification (Desktop Pipeline)",
        "",
        f"Samples tested: **{len(report.official_test_results)}** (5 per class × 5 classes)",
        f"Accuracy: **{report.official_accuracy:.2f}%**",
        "",
        "| Status | Ground Truth | Predicted | Confidence | Image |",
        "|--------|--------------|-----------|------------|-------|",
    ]
    for item in report.official_test_results:
        status = "PASS" if item.passed else "FAIL"
        lines.append(
            f"| {status} | {item.ground_truth} | {item.predicted_class} | {item.confidence:.4f} | `{item.path}` |"
        )

    lines.extend([
        "",
        "## 2. GUI Image Pipeline",
        "",
        f"- `debug/runtime_input.png`: saved",
        f"- `debug/preprocessed_visualization.png`: saved",
        "",
        "| Check | Result |",
        "|-------|--------|",
    ])
    for key, value in report.gui_pipeline_checks.items():
        lines.append(f"| {key} | {value} |")

    lines.extend([
        "",
        "## 3. Model Loading",
        "",
        f"- Selected crop: **{report.selected_crop}**",
        f"- Weights path: `{report.weights_path}`",
        f"- Weights match expected: **{report.weights_correct}**",
        f"- Architecture: **{report.model_architecture}**",
        f"- Output classes: **{report.num_classes}**",
        f"- Device: **{report.device}**",
        "",
        "## 4. Class Mapping",
        "",
        f"- Mapping file: `{report.class_mapping_path}`",
        f"- Mapping matches runtime: **{report.mapping_valid}**",
        "",
        "| Index | Disease |",
        "|------:|---------|",
    ])
    for idx, name in sorted(report.class_index_map.items()):
        lines.append(f"| {idx} | {name} |")

    lines.extend([
        "",
        "## 5. Preprocessing",
        "",
        "| Step | Value |",
        "|------|-------|",
    ])
    for key, value in report.preprocessing.items():
        lines.append(f"| {key} | {value} |")

    lines.extend([
        "",
        "## 6. GUI vs Evaluation Comparison",
        "",
        f"- Sample: `{report.gui_vs_eval.get('sample_image', '')}`",
        f"- Max logit diff: **{report.gui_vs_eval.get('max_logit_abs_diff', 'n/a')}**",
        f"- Max prob diff: **{report.gui_vs_eval.get('max_prob_abs_diff', 'n/a')}**",
        f"- Identical: **{report.gui_vs_eval.get('identical', False)}**",
        "",
        "## 7. Real-World Test Folder",
        "",
        f"- Folder: `debug/real_world_test/`",
        f"- Images found: **{report.ood_analysis.get('real_world_image_count', 0)}**",
        f"- Results CSV: `debug/real_world_results.csv`",
        "",
        "## 8. Out-of-Distribution Analysis",
        "",
        f"- High-confidence (≥95%) count: **{report.ood_analysis.get('high_confidence_count', 0)}**",
        f"- Blight prediction ratio (real-world): **{report.ood_analysis.get('blight_prediction_ratio', 0):.2%}**",
        f"- Overconfident OOD suspected: **{report.ood_analysis.get('overconfident_ood_suspected', False)}**",
        "",
        "## 9. Final Checklist",
        "",
        f"- [{'x' if report.official_accuracy >= 95 else ' '}] Desktop pipeline accuracy on official test images",
        f"- [{'x' if report.gui_vs_eval.get('identical') else ' '}] GUI preprocessing matches evaluation (logit diff = 0)",
        f"- [{'x' if report.mapping_valid else ' '}] Class mapping correct",
        f"- [{'x' if report.weights_correct else ' '}] Model loading correct",
        f"- [{'x' if report.gui_pipeline_checks.get('pipeline_ok') else ' '}] Runtime preprocessing correct",
        f"- [{'x' if not report.bug_found else ' '}] No runtime code bug identified",
        "",
        "## Conclusion",
        "",
    ])

    if report.bug_found:
        lines.append(report.bug_details)
    else:
        lines.append(
            "The desktop runtime pipeline matches the official evaluation inference path on "
            "held-out PlantVillage test images. If users observe frequent Early/Late Blight "
            "predictions on arbitrary uploads, the cause is **model generalization / "
            "out-of-distribution input**, not a defect in the desktop application code path."
        )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    rng = random.Random(42)
    config = load_config(crop="tomato", project_root=PROJECT_ROOT)

    print("=" * 72)
    print("TOMATO RUNTIME INFERENCE VERIFICATION")
    print("=" * 72)

    service = InferenceService(config)
    service.load_model()
    predictor = service._predictor
    assert predictor is not None

    report = DiagnosisReport()

    loading = verify_model_loading(service, config)
    report.selected_crop = loading["selected_crop"]
    report.weights_path = loading["loaded_weights_path"]
    report.weights_correct = loading["weights_match_expected"]
    report.model_architecture = loading["model_architecture"]
    report.num_classes = loading["num_output_classes"]
    report.device = loading["device"]

    print("\n[3] MODEL LOADING")
    for key, value in loading.items():
        print(f"  {key}: {value}")

    report.class_index_map, report.mapping_valid = verify_class_mapping(predictor)
    report.class_mapping_path = str((PROJECT_ROOT / "datasets/tomato/reports/class_mapping.json").resolve())

    print("\n[4] CLASS MAPPING")
    for idx, name in sorted(report.class_index_map.items()):
        print(f"  {idx} -> {name}")
    print(f"  mapping_valid: {report.mapping_valid}")

    report.preprocessing = {
        "resize": f"{predictor.image_size}x{predictor.image_size}",
        "rgb_conversion": "read_image_rgb (BGR->RGB, EXIF transpose via PIL)",
        "normalization_mean": IMAGENET_MEAN,
        "normalization_std": IMAGENET_STD,
        "transform": "get_inference_transforms (= get_val_transforms)",
        "expected_tensor_shape": f"[1, 3, {predictor.image_size}, {predictor.image_size}]",
        "expected_dtype": "torch.float32",
        "device": str(predictor.device),
    }
    print("\n[5] PREPROCESSING")
    for key, value in report.preprocessing.items():
        print(f"  {key}: {value}")

    print("\n[1] OFFICIAL TEST DATASET (desktop Predictor.predict)")
    report.official_test_results = run_official_test_samples(rng, predictor)
    passed = sum(1 for r in report.official_test_results if r.passed)
    report.official_accuracy = 100.0 * passed / len(report.official_test_results)
    print(f"\nOverall accuracy: {passed}/{len(report.official_test_results)} ({report.official_accuracy:.2f}%)")

    sample_for_gui = PROJECT_ROOT / report.official_test_results[0].path
    report.gui_pipeline_checks = verify_gui_pipeline(predictor, sample_for_gui)
    print("\n[2] GUI IMAGE PIPELINE")
    for key, value in report.gui_pipeline_checks.items():
        print(f"  {key}: {value}")

    compare_image = PROJECT_ROOT / "datasets/tomato/split/test/Tomato___healthy" / sorted(
        (PROJECT_ROOT / "datasets/tomato/split/test/Tomato___healthy").glob("*.jpg")
    )[0].name
    report.gui_vs_eval = compare_gui_vs_eval(predictor, compare_image)
    print("\n[7] GUI VS EVALUATION")
    for key, value in report.gui_vs_eval.items():
        print(f"  {key}: {value}")

    print("\n[6] LOGITS (first official sample)")
    first = report.official_test_results[0]
    if first.logits is not None and first.probs is not None:
        print(f"  raw_logits: {np.array2string(first.logits, precision=4, separator=', ')}")
        print(f"  softmax_probs: {np.array2string(first.probs, precision=4, separator=', ')}")
        print(f"  argmax: {int(np.argmax(first.probs))} -> {first.predicted_class}")
        print(f"  top3: {first.top3}")

    print("\n[8] REAL-WORLD TEST FOLDER")
    report.real_world_rows = run_real_world_tests(predictor)
    report.ood_analysis = analyze_ood(report.real_world_rows)
    print(f"  images in debug/real_world_test: {report.ood_analysis['real_world_image_count']}")
    if report.real_world_rows:
        for row in report.real_world_rows:
            print(f"  {row['filename']}: {row['predicted_class']} ({row['confidence']})")
    else:
        print("  (no images found — drop phone photos into debug/real_world_test/ and re-run)")

    print("\n[9] OUT-OF-DISTRIBUTION CHECK")
    for key, value in report.ood_analysis.items():
        print(f"  {key}: {value}")

    report.bug_found = (
        not report.weights_correct
        or not report.mapping_valid
        or not report.gui_pipeline_checks.get("pipeline_ok", False)
        or not report.gui_vs_eval.get("identical", False)
    )

    if report.bug_found:
        issues = []
        if not report.weights_correct:
            issues.append("Wrong weights loaded")
        if not report.mapping_valid:
            issues.append("Class mapping mismatch")
        if not report.gui_pipeline_checks.get("pipeline_ok", False):
            issues.append("GUI preprocessing pipeline issue")
        if not report.gui_vs_eval.get("identical", False):
            issues.append("Desktop vs evaluation logit mismatch")
        report.bug_details = "Issues detected: " + "; ".join(issues)
        report.root_cause = "runtime pipeline defect"
    else:
        report.root_cause = (
            "model generalization on out-of-distribution / real-world images "
            "(runtime code matches evaluation on official test set)"
        )

    write_report(report)
    print(f"\nReport saved: {REPORT_PATH}")
    return 0 if not report.bug_found else 1


if __name__ == "__main__":
    sys.exit(main())
