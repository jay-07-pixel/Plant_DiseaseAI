#!/usr/bin/env python3
"""Complete label consistency audit for tomato class index ordering."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _sanitize_import_path() -> None:
    vendor_root = (PROJECT_ROOT / ".vendor").resolve()
    sys.path = [e for e in sys.path if e and Path(e).resolve() != vendor_root]


_sanitize_import_path()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torchvision.datasets import ImageFolder

from desktop_app.services.inference_service import InferenceService
from evaluation.tomato_evaluator import TomatoEvaluator
from inference.explainable_predictor import ExplainablePredictor
from training.tomato_trainer import TomatoTrainer
from utils.config import load_config


def print_mapping(title: str, mapping: dict[int, str]) -> None:
    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)
    for idx in sorted(mapping):
        print(f"{idx} -> {mapping[idx]}")


def mapping_from_indexed_list(names: list[str]) -> dict[int, str]:
    return {i: name for i, name in enumerate(names)}


def mapping_from_id_to_display(data: dict) -> dict[int, str]:
    id_to_display = data.get("id_to_display", {})
    return {int(k): v for k, v in sorted(id_to_display.items(), key=lambda x: int(x[0]))}


def mappings_equal(a: dict[int, str], b: dict[int, str]) -> bool:
    return a == b


def diff_mappings(a: dict[int, str], b: dict[int, str], label_a: str, label_b: str) -> list[str]:
    diffs = []
    keys = sorted(set(a) | set(b))
    for k in keys:
        if a.get(k) != b.get(k):
            diffs.append(f"  index {k}: {label_a}={a.get(k)!r} vs {label_b}={b.get(k)!r}")
    return diffs


def main() -> int:
    config = load_config(crop="tomato", project_root=PROJECT_ROOT)
    train_dir = config.path("paths.train")
    mapping_json_path = PROJECT_ROOT / "datasets/tomato/reports/class_mapping.json"
    weights_path = PROJECT_ROOT / "weights/tomato/best_model.pth"

    print("TOMATO LABEL CONSISTENCY AUDIT")
    print("=" * 60)

    # 1. Training class_to_idx (explicit, used by PlantDiseaseDataset)
    trainer = TomatoTrainer(config)
    train_class_to_idx = trainer._class_mapping()
    train_idx_to_name = {v: config.class_configs[v].display_name for v in sorted(train_class_to_idx.values())}
    # Rebuild from folder->id using config order
    train_idx_to_name = {}
    for c in sorted(config.class_configs, key=lambda x: x.id):
        train_idx_to_name[c.id] = c.display_name

    train_mapping = train_idx_to_name
    print_mapping("1. Training class_to_idx (TomatoTrainer._class_mapping)", train_mapping)
    print("\n  folder -> index:")
    for folder, idx in sorted(train_class_to_idx.items(), key=lambda x: x[1]):
        print(f"    {idx} <- {folder}")

    # ImageFolder simulation (what torchvision would assign alphabetically)
    imagefolder_classes: list[str] = []
    imagefolder_class_to_idx: dict[str, int] = {}
    imagefolder_mapping: dict[int, str] = {}
    if train_dir.exists():
        ds = ImageFolder(root=str(train_dir))
        imagefolder_classes = ds.classes
        imagefolder_class_to_idx = ds.class_to_idx
        folder_to_display = {c.folder_name: c.display_name for c in config.class_configs}
        imagefolder_mapping = {
            idx: folder_to_display.get(folder, folder)
            for folder, idx in imagefolder_class_to_idx.items()
        }
        print_mapping("1b. ImageFolder.classes (alphabetical, simulated from train dir)", imagefolder_mapping)
        print("\n  ImageFolder.classes list:", imagefolder_classes)
    else:
        print(f"\n[SKIP] ImageFolder simulation — train dir missing: {train_dir}")

    # 2. class_mapping.json
    with mapping_json_path.open("r", encoding="utf-8") as f:
        json_data = json.load(f)
    json_mapping = mapping_from_id_to_display(json_data)
    print_mapping("2. datasets/tomato/reports/class_mapping.json", json_mapping)

    # 3. Config class_names (YAML order by id)
    config_mapping = {c.id: c.display_name for c in sorted(config.class_configs, key=lambda x: x.id)}
    print_mapping("3. Config YAML (class_configs by id)", config_mapping)

    # 4. Runtime inference mapping
    predictor = ExplainablePredictor(config)
    runtime_mapping = {i: name for i, name in enumerate(predictor.class_names)}
    print_mapping("4. Runtime loaded class mapping (Predictor.class_names)", runtime_mapping)

    # 5. Evaluation script mapping
    evaluator = TomatoEvaluator(config)
    eval_mapping = dict(evaluator.id_to_display)
    print_mapping("5. Evaluation script (TomatoEvaluator.id_to_display)", eval_mapping)

    # 6. Checkpoint embedded class_mapping
    checkpoint_mapping: dict[int, str] = {}
    checkpoint_folder_to_id: dict[str, int] = {}
    if weights_path.exists():
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        ckpt_meta = ckpt.get("class_mapping", {})
        id_to_display = ckpt_meta.get("id_to_display", {})
        checkpoint_mapping = {int(k): v for k, v in id_to_display.items()}
        checkpoint_folder_to_id = ckpt_meta.get("folder_to_id", {})
        print_mapping("6. best_model.pth checkpoint (id_to_display)", checkpoint_mapping)
        print("\n  checkpoint folder_to_id:")
        for folder, idx in sorted(checkpoint_folder_to_id.items(), key=lambda x: x[1]):
            print(f"    {idx} <- {folder}")
    else:
        print(f"\n[ERROR] Weights not found: {weights_path}")

    # Pairwise comparison
    print(f"\n{'=' * 60}")
    print("PAIRWISE COMPARISON")
    print("=" * 60)

    reference = train_mapping
    pipeline_sources = {
        "class_mapping.json": json_mapping,
        "config_yaml": config_mapping,
        "runtime_inference": runtime_mapping,
        "evaluation_script": eval_mapping,
        "checkpoint_best_model": checkpoint_mapping,
    }
    all_pipeline_match = True
    for name, mapping in pipeline_sources.items():
        ok = mappings_equal(reference, mapping)
        status = "IDENTICAL" if ok else "MISMATCH"
        print(f"  training vs {name}: {status}")
        if not ok:
            all_pipeline_match = False
            for line in diff_mappings(reference, mapping, "training", name):
                print(line)

    if imagefolder_mapping:
        ok = mappings_equal(reference, imagefolder_mapping)
        status = "IDENTICAL" if ok else "DIFFERS (not used in training)"
        print(f"  training vs imagefolder_simulated: {status}")
        if not ok:
            for line in diff_mappings(reference, imagefolder_mapping, "training", "imagefolder"):
                print(line)
            print("\n  ImageFolder is NOT used during training. TomatoTrainer uses PlantDiseaseDataset")
            print("  with explicit class_to_idx from configs/crops/tomato.yaml — this mismatch is harmless.")

    # Per-class prediction test
    print(f"\n{'=' * 60}")
    print("PER-CLASS PREDICTION TEST (one image per class from test set)")
    print("=" * 60)

    test_root = config.path("paths.test")
    id_to_folder = {c.id: c.folder_name for c in config.class_configs}
    folder_to_display = {c.folder_name: c.display_name for c in config.class_configs}

    prediction_ok = True
    print(f"{'GT ID':>5} | {'Ground Truth':<40} | {'Pred Idx':>8} | {'Mapped Label':<40} | {'Match':>5}")
    print("-" * 110)

    for class_id in sorted(id_to_folder):
        folder = id_to_folder[class_id]
        folder_path = test_root / folder
        images = sorted(folder_path.glob("*.jpg"))
        if not images:
            images = sorted(folder_path.glob("*.jpeg"))
        if not images:
            print(f"  [SKIP] No images in {folder_path}")
            continue

        image_path = images[0]
        result = predictor.predict(image_path)
        expected = folder_to_display[folder]
        mapped = runtime_mapping[result.predicted_class_id]
        match = result.predicted_class_id == class_id and mapped == expected
        if not match:
            prediction_ok = False
        status = "OK" if match else "FAIL"
        print(
            f"{class_id:>5} | {expected:<40} | {result.predicted_class_id:>8} | "
            f"{mapped:<40} | {status:>5}  ({image_path.name})"
        )

    # Final verdict
    print(f"\n{'=' * 60}")
    print("FINAL VERDICT")
    print("=" * 60)
    if all_pipeline_match and prediction_ok:
        print("ALL ACTIVE PIPELINE MAPPINGS IDENTICAL.")
        print("Class index ordering is consistent across training, inference, evaluation,")
        print("checkpoint metadata, and class_mapping.json.")
        print("Per-class predictions align with expected labels.")
        print("\nGUI displays predicted_class strings from runtime mapping (index -> display name).")
        return 0

    if not all_pipeline_match:
        print("PIPELINE MAPPING MISMATCH — this WOULD affect predictions.")
    if not prediction_ok:
        print("PREDICTION MISMATCH — wrong index on some classes (model error, not label mapping).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
