# Architecture

**Analysis Date:** 2026-03-04

## Pattern Overview

**Overall:** Linear Pipeline Architecture (ML Training & Deployment)

**Key Characteristics:**
- Sequential workflow: Download → Train → Export → Infer → Deploy
- Single-model approach with YOLO26 architecture
- Script-based orchestration (no complex framework)
- Model-agnostic inference layer

## Layers

**Data Layer:**
- Purpose: Dataset management and storage
- Location: `dataset/` (downloaded from Roboflow)
- Contains: Training/validation images and YOLO-format labels
- Depends on: Roboflow API
- Used by: `train_yolo.py`, `val_metrics.py`

**Model Layer:**
- Purpose: Model training, export, and inference
- Location: `src/train_yolo.py`, `src/export_model.py`, `src/inferencia.py`
- Contains: YOLO model operations using Ultralytics library
- Depends on: Ultralytics library
- Used by: All scripts that work with predictions

**API Layer:**
- Purpose: Real-time inference service
- Location: `src/main_api.py`
- Contains: FastAPI application with /predict endpoint
- Depends on: FastAPI, Uvicorn
- Used by: External clients via HTTP POST

**Utility Layer:**
- Purpose: Shared helper functions
- Location: `src/utils.py`
- Contains: Placeholder functions (convert_format, visualize_dataset)
- Depends on: None
- Used by: Other modules (currently unused)

## Data Flow

**Training Pipeline:**

1. `download_dataset.py` → Downloads labeled dataset from Roboflow to `dataset/`
2. `train_yolo.py` → Trains YOLO26m model using `dataset/data.yaml` config
3. Output saved to `runs/detect/train_casas/weights/best.pt`

**Inference Pipeline:**

1. `inferencia.py` → Loads best model (local or from Colab)
2. Runs predictions on validation images
3. Displays and saves annotated results

**Production Pipeline:**

1. `export_model.py` → Converts best.pt to ONNX format
2. Output saved to `models/house_detector_prod.onnx`
3. `main_api.py` → Loads ONNX/PT model and serves predictions

**Validation Pipeline:**

1. `val_metrics.py` → Evaluates model on validation set
2. Reports mAP@0.5, Precision, Recall

## Key Abstractions

**Model Loader:**
- Purpose: Find best trained model
- Examples: `find_best_model()` function in `src/inferencia.py`, `src/export_model.py`, `src/val_metrics.py`, `src/main_api.py`
- Pattern: Glob search for `runs/detect/train*/weights/best.pt`, fallback to `models/best_colab.pt`, fallback to base model

**Config Loader:**
- Purpose: Load dataset configuration
- Examples: `dataset/data.yaml` (YOLO format)
- Pattern: Standard YOLO dataset configuration with train/val/test paths

## Entry Points

**Training Entry:**
- Location: `src/train_yolo.py`
- Triggers: `python src/train_yolo.py`
- Responsibilities: Initialize YOLO26m, configure training, execute training loop, report metrics

**API Entry:**
- Location: `src/main_api.py`
- Triggers: `python src/main_api.py` or `uvicorn src.main_api:app`
- Responsibilities: Start FastAPI server, load model on startup, handle /predict POST requests

**Dataset Download Entry:**
- Location: `src/download_dataset.py`
- Triggers: `python src/download_dataset.py`
- Responsibilities: Authenticate with Roboflow, download YOLO-format dataset

**Inference Entry:**
- Location: `src/inferencia.py`
- Triggers: `python src/inferencia.py`
- Responsibilities: Find model, run inference on validation image, display and save results

**Export Entry:**
- Location: `src/export_model.py`
- Triggers: `python src/export_model.py`
- Responsibilities: Find best weights, export to ONNX format, save to models/

**Validation Entry:**
- Location: `src/val_metrics.py`
- Triggers: `python src/val_metrics.py`
- Responsibilities: Find model, run validation, report metrics

## Error Handling

**Strategy:** Basic exception handling with user-friendly messages

**Patterns:**
- API: FastAPI HTTPException for validation errors, try-catch for processing errors
- Scripts: Early return with error message if prerequisites not met
- Model loading: Fallback chain (local → Colab → base model)

## Cross-Cutting Concerns

**Logging:** Print statements with separators and formatted output

**Configuration:** Environment variables via python-dotenv (.env file)

**Model Management:** Version-based (weights stored with timestamps, best.pt convention)

---

*Architecture analysis: 2026-03-04*
