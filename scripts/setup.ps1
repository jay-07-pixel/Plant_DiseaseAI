# One-command setup after cloning Plant_DiseaseAI (Windows PowerShell)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Write-Host "=== PlantDiseaseAI setup ==="

$grape = "weights\grape\best_model.pth"
$tomato = "weights\tomato\best_model.pth"
if (-not (Test-Path $grape) -or -not (Test-Path $tomato)) {
    Write-Error "Model weights missing. Re-clone or restore weights/grape and weights/tomato."
}

python -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip wheel
python -m pip install -r requirements.txt

if (-not (Test-Path ".env") -and (Test-Path ".env.example")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env from .env.example — add GROQ_API_KEY for AI tips (optional)."
}

Write-Host ""
Write-Host "Setup complete. Run:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  python scripts/run_app.py"
