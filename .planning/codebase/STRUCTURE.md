# Codebase Structure

**Analysis Date:** 2026-03-04

## Directory Layout

```
taller-yolo-deteccion-casas/
├── src/                      # Source code (main application scripts)
│   ├── train_yolo.py         # Model training script
│   ├── export_model.py       # Model export to ONNX
│   ├── inferencia.py         # Inference/prediction script
│   ├── download_dataset.py   # Dataset download from Roboflow
│   ├── val_metrics.py        # Validation metrics reporting
│   ├── main_api.py           # FastAPI server
│   └── utils.py              # Utility functions (placeholder)
├── models/                   # Production models (generated)
│   └── house_detector_prod.onnx  # Exported ONNX model
├── runs/                     # Training outputs (generated, gitignored)
│   └── detect/
│       └── train_casas/      # Training run directory
│           ├── weights/      # Model checkpoints
│           │   ├── best.pt
│           │   └── last.pt
│           ├── results.csv  # Training metrics over epochs
│           └── *.png         # Visualization plots
├── dataset/                  # Downloaded dataset (gitignored)
│   ├── train/images/
│   ├── train/labels/
│   ├── valid/images/
│   ├── valid/labels/
│   └── data.yaml             # Dataset configuration
├── .env                      # Environment variables (gitignored)
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── .gitignore                # Git ignore rules
```

## Directory Purposes

**src/:**
- Purpose: Main application code
- Contains: Python scripts for training, inference, export, API
- Key files: `train_yolo.py`, `main_api.py`, `inferencia.py`

**models/:**
- Purpose: Production-ready exported models
- Contains: ONNX format model files
- Key files: `house_detector_prod.onnx`

**runs/:**
- Purpose: Training artifacts and outputs
- Contains: Weights, metrics, visualizations
- Generated: Yes (created during training)
- Committed: No (gitignored)

**dataset/:**
- Purpose: Training and validation data
- Contains: Images and YOLO-format label files
- Generated: Yes (downloaded from Roboflow)
- Committed: No (gitignored)

## Key File Locations

**Entry Points:**
- `src/train_yolo.py`: Start model training
- `src/main_api.py`: Start FastAPI server
- `src/inferencia.py`: Run inference on images
- `src/download_dataset.py`: Download dataset
- `src/export_model.py`: Export to production format
- `src/val_metrics.py`: Generate validation metrics

**Configuration:**
- `.env`: API keys and environment variables
- `requirements.txt`: Python dependencies
- `dataset/data.yaml`: YOLO dataset configuration (after download)

**Core Logic:**
- `src/train_yolo.py`: YOLO model training (lines 1-30)
- `src/main_api.py`: FastAPI prediction endpoint (lines 34-58)
- `src/inferencia.py`: Model inference logic (lines 19-32)

**Testing:**
- No dedicated test directory. Tests appear to be ad-hoc via `val_metrics.py` and `inferencia.py`

## Naming Conventions

**Files:**
- Python scripts: snake_case (e.g., `train_yolo.py`, `download_dataset.py`)
- Configuration: snake_case (e.g., `data.yaml`, `.env`)
- Models: descriptive with format extension (e.g., `house_detector_prod.onnx`)

**Directories:**
- Generic directories: lowercase (e.g., `src/`, `models/`, `dataset/`)
- Training runs: `train_casas` prefix with optional suffix

**Functions:**
- snake_case: `find_best_model()`, `download()`, `train()`, `export()`

## Where to Add New Code

**New Feature (Script):**
- Primary code: `src/<feature_name>.py`
- Entry point: Add to README.md command table

**New Utility Function:**
- Shared helpers: `src/utils.py`
- Note: Currently unused, placeholder functions exist

**New Model Variant:**
- Weights: `models/best_<variant>.pt` (e.g., `models/best_colab.pt`)
- Production: `models/house_detector_prod.onnx`

**Dataset Changes:**
- Configuration: Update `.env` with Roboflow credentials
- Re-download: Run `python src/download_dataset.py`

## Special Directories

**runs/detect/:**
- Purpose: YOLO training outputs
- Generated: Yes (by Ultralytics during training)
- Committed: No (in .gitignore)

**dataset/:**
- Purpose: Downloaded training data
- Generated: Yes (by download_dataset.py)
- Committed: No (in .gitignore)

**models/:**
- Purpose: Production model storage
- Generated: Yes (by export_model.py)
- Committed: Partially (ONNX only, not .pt files per .gitignore)

**venv/:**
- Purpose: Python virtual environment
- Generated: Yes
- Committed: No (in .gitignore)

---

*Structure analysis: 2026-03-04*
