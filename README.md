# PlantDiseaseAI

Production-grade **offline desktop application** for multi-crop plant leaf disease detection using deep learning, Grad-CAM explainability, and optional Groq-powered farmer guidance.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red)
![License](https://img.shields.io/badge/License-MIT-green)

## Features

- **Multi-crop support** — Grape (4 classes) and Tomato (10 classes)
- **EfficientNet-B0** inference with top-3 predictions and confidence scores
- **Grad-CAM overlays** — visual explanation of model focus regions
- **Modern desktop UI** — PySide6 app with upload, webcam/Pi camera capture, and multilingual UI (English, Hindi, Marathi)
- **AI recommendations** — Prevention, remedies, and tips via Groq LLM (optional, requires API key)
- **Windows `.exe` build** — PyInstaller packaging for distribution
- **Raspberry Pi ready** — CPU mode, picamera2 + OpenCV camera backends

## Supported Crops

| Crop   | Classes | Input Size | Model            | Status        |
|--------|---------|------------|------------------|---------------|
| Grape  | 4       | 224×224    | EfficientNet-B0  | Production    |
| Tomato | 10      | 256×256    | EfficientNet-B0  | Production    |
| Potato | 3       | —          | —                | Audit only    |

> **Clone & run:** Model weights and class mappings are included in this repo so the desktop app works immediately after setup.

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/jay-07-pixel/Plant_DiseaseAI.git
cd Plant_DiseaseAI
```

**Windows (PowerShell):**
```powershell
.\scripts\setup.ps1
python scripts\run_app.py
```

**Linux / macOS / Raspberry Pi:**
```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
python scripts/run_app.py
```

**Manual install:**
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/run_app.py
```

### 2. Configure environment (optional — for AI tips)

Copy `.env.example` to `.env` and set `GROQ_API_KEY`.

### 3. Run with a specific crop

```bash
python scripts/run_app.py --crop tomato
```

## Windows Executable

Build and copy to Desktop:

```powershell
python scripts/build_exe.py
```

Output folder: `dist/PlantDiseaseAI/` (~4.6 GB with PyTorch + both models).

> **Important:** Distribute the **entire folder**, not only `PlantDiseaseAI.exe`. The executable depends on `_internal/`, `configs/`, `weights/`, and `resources/`.

## Raspberry Pi

```bash
chmod +x scripts/setup_raspberry_pi.sh
./scripts/setup_raspberry_pi.sh
python scripts/test_pi_camera.py
python scripts/run_app.py
```

Pi mode auto-enables CPU inference and disables Grad-CAM by default. See [ARCHITECTURE.md](ARCHITECTURE.md) for platform details.

## Project Structure

```
Plant_DiseaseAI/
├── configs/           # YAML configuration (base + crop + platform)
├── desktop_app/       # PySide6 UI, controllers, services
├── inference/         # Predictor, Grad-CAM, explainable inference
├── training/          # Training loop, datasets, metrics
├── preprocessing/     # Dataset audit, split, balance pipelines
├── evaluation/        # Test-set evaluation and plots
├── scripts/           # CLI entry points and tooling
├── resources/         # Translations and app icon
├── weights/           # Trained model checkpoints (local, gitignored)
├── reports/           # Generated audit and training reports
└── tests/             # Pytest suite
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for a full system design overview.

## Training Pipeline (per crop)

```bash
# Grape — full pipeline
python scripts/run_train_eval_export.py

# Tomato — step-by-step
python scripts/audit_tomato_dataset.py
python scripts/preprocess_tomato_raw.py
python scripts/split_tomato_dataset.py
python scripts/balance_tomato_train.py
python scripts/train_tomato.py
python scripts/evaluate_tomato.py
```

## Scripts Reference

| Script | Purpose |
|--------|---------|
| `scripts/run_app.py` | Launch desktop application |
| `scripts/build_exe.py` | Build Windows executable |
| `scripts/setup_raspberry_pi.sh` | Pi dependency setup |
| `scripts/test_pi_camera.py` | Camera diagnostic on Pi |
| `scripts/run_gradcam_demo.py` | Grad-CAM demo |
| `scripts/generate_confusion_matrices.py` | Confusion matrix plots |

## Configuration

Configs merge in order: `configs/base.yaml` → `configs/crops/{crop}.yaml` → `configs/platform/raspberry_pi.yaml` (on Pi).

Key settings:

- **Device:** `device.preferred` — `cuda` | `cpu` | `mps`
- **Camera:** `desktop_app.camera.backend` — `auto` | `opencv` | `picamera2`
- **Grad-CAM:** `inference.enable_gradcam` or `PLANT_DISEASE_ENABLE_GRADCAM=1`

## Development

```bash
pytest tests/
ruff check .
```

## License

MIT License — see project license terms.

## Author

[jay-07-pixel](https://github.com/jay-07-pixel)
