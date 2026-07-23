#!/usr/bin/env python3
"""End-to-end audit: Tomato desktop inference vs training/evaluation pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _sanitize_import_path() -> None:
    vendor_root = (PROJECT_ROOT / ".vendor").resolve()
    sys.path = [
        entry for entry in sys.path if entry and Path(entry).resolve() != vendor_root
    ]


_sanitize_import_path()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from inference.predictor import Predictor
from models.factory import ModelFactory
from training.dataset import PlantDiseaseDataset
from training.transforms import IMAGENET_MEAN, IMAGENET_STD, get_inference_transforms, get_val_transforms
from utils.config import load_config


def _find_sample_image(test_root: Path, folder: str) -> Path:
    class_dir = test_root / folder
    for path in sorted(class_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            return path
    raise FileNotFoundError(f"No sample image in {class_dir}")


def _eval_forward(config, image_path: Path, device: torch.device) -> dict:
    class_mapping = {c.folder_name: c.id for c in config.class_configs}
    image_size = int(config.get("inference.image_size", config.get("training.image_size", 256)))
    dataset = PlantDiseaseDataset(
        root=config.path("paths.test"),
        class_to_idx=class_mapping,
        transform=get_val_transforms(image_size),
    )
    sample_idx = next(
        i for i, (path, _) in enumerate(dataset.samples) if path.resolve() == image_path.resolve()
    )
    tensor, label = dataset[sample_idx]

    weights_path = config.project_root / config.get("inference.weights_path")
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    class_meta = checkpoint.get("class_mapping", {})
    model_name = checkpoint.get("model_name") or class_meta.get("model_name", "efficientnet_b0")

    mapping_path = config.project_root / config.get("inference.class_names_path")
    with mapping_path.open("r", encoding="utf-8") as handle:
        mapping_data = json.load(handle)

    model = ModelFactory.create(
        model_name=str(model_name),
        num_classes=len(mapping_data.get("id_to_display", config.class_names)),
        pretrained=False,
        device=device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        batch = tensor.unsqueeze(0).to(device)
        outputs = model(batch)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = int(torch.argmax(probs, dim=1).item())
        top3_idx = torch.topk(probs, k=3, dim=1).indices[0].cpu().tolist()
        top3_conf = torch.topk(probs, k=3, dim=1).values[0].cpu().tolist()

    id_to_display = mapping_data["id_to_display"]
    sorted_names = [id_to_display[str(i)] for i in range(len(id_to_display))]

    return {
        "pipeline": "evaluation",
        "ground_truth_index": label,
        "weights_path": str(weights_path),
        "class_mapping_path": str(mapping_path),
        "image_size": image_size,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "transform": "get_val_transforms (Resize + Normalize + ToTensorV2)",
        "rgb_conversion": "read_image_rgb via dataset",
        "logits": outputs[0].cpu().numpy(),
        "probs": probs[0].cpu().numpy(),
        "pred_idx": pred_idx,
        "pred_name": sorted_names[pred_idx],
        "top3": list(zip(top3_idx, top3_conf, [sorted_names[i] for i in top3_idx])),
    }


def _desktop_forward(config, image_path: Path) -> dict:
    predictor = Predictor(config)
    tensor = predictor._preprocess(image_path)

    with torch.no_grad():
        logits_t, probs_t = predictor._infer_pytorch(tensor)
        pred_idx = int(torch.argmax(probs_t, dim=1).item())
        top3 = torch.topk(probs_t, k=3, dim=1)

    result = predictor.predict(image_path)

    return {
        "pipeline": "desktop",
        "weights_path": str(predictor.weights_path),
        "class_mapping_path": str(predictor.class_names_path),
        "image_size": predictor.image_size,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "transform": "get_inference_transforms (= get_val_transforms)",
        "rgb_conversion": "read_image_rgb in _preprocess",
        "logits": logits_t[0].cpu().numpy(),
        "probs": probs_t[0].cpu().numpy(),
        "pred_idx": pred_idx,
        "pred_name": predictor.id_to_name[pred_idx],
        "top3": [
            (int(idx), float(conf), predictor.id_to_name[int(idx)])
            for idx, conf in zip(top3.indices[0].tolist(), top3.values[0].tolist())
        ],
        "result_top3": [(p.class_id, p.confidence, p.class_name) for p in result.top_predictions],
    }


def _compare(eval_result: dict, desktop_result: dict) -> dict:
    logits_diff = float(np.max(np.abs(eval_result["logits"] - desktop_result["logits"])))
    probs_diff = float(np.max(np.abs(eval_result["probs"] - desktop_result["probs"])))
    return {
        "weights_match": eval_result["weights_path"] == desktop_result["weights_path"],
        "mapping_match": eval_result["class_mapping_path"] == desktop_result["class_mapping_path"],
        "image_size_match": eval_result["image_size"] == desktop_result["image_size"],
        "pred_idx_match": eval_result["pred_idx"] == desktop_result["pred_idx"],
        "pred_name_match": eval_result["pred_name"] == desktop_result["pred_name"],
        "top3_idx_match": [t[0] for t in eval_result["top3"]] == [t[0] for t in desktop_result["top3"]],
        "max_logits_diff": logits_diff,
        "max_probs_diff": probs_diff,
        "pipelines_aligned": (
            eval_result["pred_idx"] == desktop_result["pred_idx"]
            and logits_diff < 1e-5
            and probs_diff < 1e-6
        ),
    }


def main() -> int:
    config = load_config(crop="tomato", project_root=PROJECT_ROOT)
    test_root = config.path("paths.test")
    sample = _find_sample_image(test_root, "Tomato___healthy")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Tomato Inference Pipeline Audit")
    print("=" * 60)
    print(f"Sample image: {sample}")
    print(f"Device: {device}")
    print()

    eval_result = _eval_forward(config, sample, device)
    desktop_result = _desktop_forward(config, sample)
    comparison = _compare(eval_result, desktop_result)

    for label, result in ("EVALUATION", eval_result), ("DESKTOP", desktop_result):
        print(label)
        print(f"  weights:         {result['weights_path']}")
        print(f"  class mapping:   {result['class_mapping_path']}")
        print(f"  image size:      {result['image_size']}")
        print(f"  preprocessing:   {result['transform']}")
        print(f"  RGB:             {result['rgb_conversion']}")
        print(f"  mean/std:        {result['mean']} / {result['std']}")
        print(f"  pred index/name: {result['pred_idx']} / {result['pred_name']}")
        print(f"  top-3:           {result['top3']}")
        print()

    print("COMPARISON")
    for key, value in comparison.items():
        print(f"  {key}: {value}")

    report = {
        "crop": "tomato",
        "sample_image": str(sample.relative_to(PROJECT_ROOT)),
        "evaluation": {
            k: v for k, v in eval_result.items() if k not in {"logits", "probs"}
        },
        "desktop": {
            k: v for k, v in desktop_result.items() if k not in {"logits", "probs"}
        },
        "comparison": comparison,
    }
    report_path = PROJECT_ROOT / "reports" / "tomato_inference_audit.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport saved: {report_path}")

    return 0 if comparison["pipelines_aligned"] else 1


if __name__ == "__main__":
    sys.exit(main())
