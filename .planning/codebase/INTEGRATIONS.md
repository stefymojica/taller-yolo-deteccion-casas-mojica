# External Integrations

**Analysis Date:** 2026-03-04

## APIs & External Services

**Computer Vision / ML:**
- Roboflow - Dataset management and hosting
  - SDK/Client: roboflow Python package
  - Purpose: Download labeled dataset for house detection training
  - Auth: ROBOFLOW_API_KEY (from .env)
  - Workspace/Project: Configured via ROBOFLOW_WORKSPACE, ROBOFLOW_PROJECT env vars
  - Version: Configurable via ROBOFLOW_VERSION (default: 1)

**Model Training:**
- Ultralytics HUB (implied) - Model registry
  - Purpose: Loading pre-trained YOLO weights (yolo26m.pt)
  - Model: YOLO11 Medium (yolo26m.pt) - downloaded automatically by ultralytics

**Optional:**
- Google Colab - Cloud training
  - Purpose: Alternative to local training for GPU acceleration
  - Integration: Manual download of best_colab.pt to models/ directory

## Data Storage

**Datasets:**
- Roboflow hosted dataset
  - Connection: Roboflow API
  - Format: YOLO format (images + label txt files)
  - Downloaded to: `./dataset/` directory

**Model Storage:**
- Local filesystem
  - Trained models: `runs/detect/train*/weights/best.pt`
  - Production model: `models/house_detector_prod.onnx`
  - Pre-trained weights: `yolo26m.pt` (auto-downloaded)

**Dataset Structure:**
```
dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── data.yaml
```

## Authentication & Identity

**Roboflow Authentication:**
- API Key-based authentication
- Implementation: api_key passed to Roboflow() client
- Environment variable: ROBOFLOW_API_KEY

**FastAPI:**
- No authentication currently implemented
- API is publicly accessible

## Monitoring & Observability

**Training Metrics:**
- Ultralytics built-in logging
- Output: Console output with mAP, precision, recall metrics
- Saved to: runs/detect/train*/results.csv

**Validation:**
- Script: `src/val_metrics.py`
- Metrics reported: mAP@0.5, Precision, Recall

**Error Tracking:**
- None implemented
- Errors handled via try/except and returned as HTTP 500 in API

## CI/CD & Deployment

**Hosting:**
- Local deployment (development)
- Production: Any ASGI-compatible host (uvicorn, gunicorn)

**CI Pipeline:**
- None configured

**Model Export:**
- ONNX format for production deployment
- Export script: `src/export_model.py`
- Target format: ONNX (opset=17, dynamic=True)

## Environment Configuration

**Required env vars:**
- ROBOFLOW_API_KEY - Roboflow authentication
- ROBOFLOW_WORKSPACE - Roboflow workspace name
- ROBOFLOW_PROJECT - Roboflow project name
- ROBOFLOW_VERSION - Dataset version number (optional, default: 1)

**Secrets location:**
- `.env` file in project root (excluded from git)

## Webhooks & Callbacks

**Incoming:**
- FastAPI `/predict` endpoint - Accepts image file uploads via POST
- Content-Type: multipart/form-data with image file

**Outgoing:**
- None configured

---

*Integration audit: 2026-03-04*
