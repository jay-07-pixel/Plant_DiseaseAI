#!/usr/bin/env bash
# One-command setup after cloning Plant_DiseaseAI
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "=== PlantDiseaseAI setup ==="

if [ ! -f "weights/grape/best_model.pth" ] || [ ! -f "weights/tomato/best_model.pth" ]; then
  echo "ERROR: Model weights missing. Re-clone or restore weights/grape and weights/tomato."
  exit 1
fi

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install -r requirements.txt

if [ ! -f ".env" ] && [ -f ".env.example" ]; then
  cp .env.example .env
  echo "Created .env from .env.example — add GROQ_API_KEY for AI tips (optional)."
fi

echo ""
echo "Setup complete. Run:"
echo "  source .venv/bin/activate"
echo "  python scripts/run_app.py"
