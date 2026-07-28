# PlantDiseaseAI — Junior Developer Workplan

**Project:** Multi-crop plant leaf disease detection (desktop + Raspberry Pi)  
**Repo:** https://github.com/jay-07-pixel/Plant_DiseaseAI  
**Stack:** Python 3.10+, PyTorch, EfficientNet-B0, PySide6, Grad-CAM, optional Groq LLM  
**Docs to read first:** `README.md`, `ARCHITECTURE.md`

---

## 1. Goal

Build and maintain a **production-ready offline desktop app** that:

1. Detects leaf diseases for **Grape** and **Tomato**
2. Shows **top-3 predictions** with confidence
3. Explains model focus with **Grad-CAM** (where enabled)
4. Optionally shows farmer guidance via **Groq** (Prevention / Remedies / Tips)
5. Runs on **Windows** (`.exe`) and **Raspberry Pi** (camera + CPU)

---

## 2. Current Project Status (what already exists)

| Area | Status | Notes |
|------|--------|-------|
| Grape model + app path | **Done** | 4 classes, weights in repo |
| Tomato model + app path | **Done** | 10 classes, weights in repo |
| Desktop UI (PySide6) | **Done** | Upload, capture, multilingual UI |
| Grad-CAM | **Done** | Off by default on Pi (CPU-heavy) |
| Groq AI tips | **Done (optional)** | Needs `.env` + internet |
| Windows `.exe` build | **Done** | `scripts/build_exe.py` |
| Raspberry Pi camera | **Done** | picamera2 + OpenCV backends |
| Potato pipeline | **Audit only** | Not trained / not in app yet |

**Do not rebuild everything from scratch.** Extend the existing modular design (crop YAML + shared inference/UI).

---

## 3. Week-by-Week Workplan

### Phase 0 — Environment & First Run (Day 1–2)

**Objective:** Clone, install, run the app successfully.

| # | Task | Done when |
|---|------|-----------|
| 0.1 | Clone repo and create venv | `python scripts/run_app.py` opens UI |
| 0.2 | Confirm grape + tomato weights exist | `weights/grape/best_model.pth`, `weights/tomato/best_model.pth` |
| 0.3 | Run sample inference via UI (Upload) | Prediction + confidence appear |
| 0.4 | Optional: add Groq key | Tips panel fills after prediction |
| 0.5 | Read `ARCHITECTURE.md` | Can explain UI → Inference → Model flow |

**Setup (Windows):**
```powershell
git clone https://github.com/jay-07-pixel/Plant_DiseaseAI.git
cd Plant_DiseaseAI
.\scripts\setup.ps1
python scripts\run_app.py
```

**Groq (optional):**
```text
Copy .env.example → .env
Set GROQ_API_KEY=gsk_...
```

---

### Phase 1 — Understand Codebase (Day 3–5)

**Objective:** Map folders to responsibilities. No large code changes yet.

| Folder | What to study | Focus questions |
|--------|---------------|-----------------|
| `desktop_app/` | UI + controller + services | How upload/capture triggers inference? |
| `inference/` | Predictor + Grad-CAM | How RGB image becomes class + heatmap? |
| `configs/` | Base + crop YAML | What changes when crop switches? |
| `training/` | Trainer + transforms | How train/val transforms differ? |
| `preprocessing/` | Audit / split / balance | Why class balancing is needed? |
| `scripts/` | Entry points | Which script for train / eval / app / exe? |

**Deliverable:** Short notes (1–2 pages) covering:

1. End-to-end flow for one uploaded image  
2. How grape vs tomato configs differ  
3. Where weights and class mappings are loaded from  

---

### Phase 2 — Dataset & Training Pipeline Practice (Week 2)

**Objective:** Reproduce (or partially reproduce) the tomato pipeline so you understand data quality gates.

| Step | Script / area | Output |
|------|---------------|--------|
| 2.1 | Dataset audit | Class counts, issues, report |
| 2.2 | Preprocess / clean | Valid RGB images, duplicates removed |
| 2.3 | Train/val/test split | Balanced split policy |
| 2.4 | Train-set balancing | Augmented minority classes |
| 2.5 | Train EfficientNet-B0 | `best_model.pth` + metrics |
| 2.6 | Evaluate | Accuracy, F1, confusion matrix |

**Tomato reference scripts:**
```bash
python scripts/audit_tomato_dataset.py
python scripts/preprocess_tomato_raw.py
python scripts/split_tomato_dataset.py
python scripts/balance_tomato_train.py
python scripts/train_tomato.py
python scripts/evaluate_tomato.py
```

**Rules:**

- Always write / update a report under `reports/`
- Never overwrite production weights without backup
- Keep train/val/test leakage checks documented

---

### Phase 3 — Inference, Grad-CAM & UI Features (Week 3)

**Objective:** Improve or harden the product path (not only training).

| # | Task | Acceptance criteria |
|---|------|---------------------|
| 3.1 | Verify grape + tomato inference match training preprocessing | Same image size / normalize / RGB order |
| 3.2 | Grad-CAM demo + UI overlay | Overlay looks aligned with leaf lesions |
| 3.3 | Multilingual UI check (en / hi / mr) | Labels switch; Groq language follows UI language |
| 3.4 | Camera capture (webcam) | Capture dialog shows live preview and saves frame |
| 3.5 | Error handling | Missing weights / bad image shows clear UI message |

**Useful scripts:**
```bash
python scripts/run_gradcam_demo.py
python scripts/verify_tomato_runtime.py
```

---

### Phase 4 — Packaging & Deployment (Week 4)

#### A) Windows executable

| # | Task | Notes |
|---|------|-------|
| 4.1 | Build with `python scripts/build_exe.py` | Takes 10–15+ minutes |
| 4.2 | Deploy full folder to Desktop | **Never ship only the `.exe`** |
| 4.3 | Add `.env` next to exe for Groq | Same folder as `PlantDiseaseAI.exe` |
| 4.4 | Smoke test | Open app → upload → predict → tips |

**Known packaging pitfalls:**

- Do **not** exclude runtime deps used by albumentations (e.g. `scipy`)
- Distribute entire `PlantDiseaseAI/` folder (`_internal/`, `weights/`, `configs/`)
- App icon: `resources/app_icon.ico`

#### B) Raspberry Pi

| # | Task | Notes |
|---|------|-------|
| 4.5 | Put project on external storage if SD card is small | Pendrive OK for code/weights |
| 4.6 | Create venv on a Linux/NTFS partition (**not FAT32 USB**) | FAT cannot create venv symlinks |
| 4.7 | Install torch CPU + `requirements-raspberry-pi.txt` | Large downloads; use retries |
| 4.8 | Install/use `picamera2` (system package) | App appends system packages safely |
| 4.9 | Test camera | `python scripts/test_pi_camera.py` |
| 4.10 | Run app | `python scripts/run_app.py` |

**Pi daily start (after reboot):**
```bash
sudo mount /dev/mmcblk0p3 "/media/smartlabsu/New Volume"   # if needed
source "/media/smartlabsu/New Volume/plantdisease-venv/bin/activate"
cd "/path/to/Plant_DiseaseAI"
python scripts/run_app.py
```

**Pi pitfalls:**

- Root `/` full → desktop login may fail; free space with `apt clean`
- CRLF in `.sh` scripts → use LF (repo has `.gitattributes`)
- Do **not** set `PYTHONPATH=/usr/lib/python3/dist-packages` (breaks typing/pydantic)
- Grad-CAM is disabled by default on Pi for speed

---

### Phase 5 — New Crop Extension: Potato (Stretch / Next Sprint)

**Objective:** Follow the same modular pattern used for tomato.

| Step | Work |
|------|------|
| 5.1 | Complete potato dataset audit (already started) |
| 5.2 | Preprocess + split + balance |
| 5.3 | Add `configs/crops/potato.yaml` |
| 5.4 | Train EfficientNet-B0 for potato classes |
| 5.5 | Wire weights + class mapping into app |
| 5.6 | UI crop dropdown + translations |
| 5.7 | Evaluate and write `reports/potato_*.md` |

**Design rule:** Prefer config + data over hard-coding class lists in UI.

---

## 4. Definition of Done (per feature)

A task is done only when **all** apply:

1. Code runs locally without crash  
2. Behavior matches acceptance criteria  
3. Short note / report updated if data or metrics changed  
4. No secrets committed (`.env` stays gitignored)  
5. Demoable to mentor in &lt; 5 minutes  

---

## 5. Coding Standards

- Follow existing folder layout; do not invent parallel apps  
- Prefer small PRs / commits with clear messages  
- Keep crop-specific logic in `configs/crops/*.yaml` and crop scripts  
- Use existing services (`InferenceService`, `CameraService`, `GroqExplanationService`)  
- Add tests under `tests/` for non-UI logic when practical  
- Never commit API keys, large raw datasets, or `dist/` build folders  

---

## 6. Suggested Daily Routine

1. Pull latest `main`  
2. Activate venv  
3. Pick **one** workplan item  
4. Implement + smoke test  
5. Write 3–5 bullet progress update  
6. Ask for review before changing production weights  

---

## 7. Demo Checklist (End of Sprint)

- [ ] App launches on Windows (source or `.exe`)  
- [ ] Grape prediction works  
- [ ] Tomato prediction works  
- [ ] Grad-CAM overlay visible (Windows)  
- [ ] Capture works (webcam or Pi camera)  
- [ ] Language switch works (en/hi/mr)  
- [ ] Groq tips work with `.env` configured  
- [ ] Mentors can run from README without tribal knowledge  

---

## 8. Mentorship Checkpoints

| Checkpoint | When | What mentor reviews |
|------------|------|---------------------|
| CP1 | End Phase 0 | App runs; junior understands setup |
| CP2 | End Phase 1 | Architecture notes |
| CP3 | End Phase 2 | Training report + metrics |
| CP4 | End Phase 3 | UI/inference demo |
| CP5 | End Phase 4 | `.exe` and/or Pi demo |
| CP6 | Phase 5 | Potato plan + first audit/train results |

---

## 9. Quick Reference — Important Paths

| Path | Purpose |
|------|---------|
| `scripts/run_app.py` | Launch desktop app |
| `scripts/build_exe.py` | Windows build + Desktop deploy |
| `scripts/setup_raspberry_pi.sh` | Pi setup helper |
| `scripts/test_pi_camera.py` | Camera diagnostic |
| `configs/crops/grape.yaml` | Grape model/runtime config |
| `configs/crops/tomato.yaml` | Tomato model/runtime config |
| `configs/platform/raspberry_pi.yaml` | Pi overrides (CPU, Grad-CAM off) |
| `weights/{crop}/best_model.pth` | Trained checkpoints |
| `datasets/{crop}/reports/class_mapping.json` | Class index → name |
| `.env` | `GROQ_API_KEY` only (local, not committed) |
| `ARCHITECTURE.md` | System design |
| `README.md` | User-facing setup |

---

## 10. Ownership Suggestion (Junior vs Mentor)

| Owner | Responsibility |
|-------|----------------|
| **Junior** | Setup, docs notes, dataset scripts practice, UI smoke tests, Pi/Windows packaging practice, potato audit/extension |
| **Mentor** | Architecture decisions, production weight approval, release cuts, Groq key policy, hardware issues on lab Pi |

---

## 11. First 3 Tasks for Junior (start tomorrow)

1. Clone repo → run app → predict one grape image and one tomato image  
2. Read `ARCHITECTURE.md` and write a 1-page flow diagram in your own words  
3. Configure `.env` for Groq and verify Tips panel works  

After that, ask mentor which track next: **Training (Phase 2)**, **UI/Inference (Phase 3)**, or **Deployment (Phase 4)**.
