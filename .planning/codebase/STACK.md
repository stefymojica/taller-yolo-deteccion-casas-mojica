# Technology Stack

**Analysis Date:** 2026-03-04

## Languages

**Primary:**
- Python 3.x - Core language for all scripts, training, inference, and API

**Secondary:**
- Not applicable

## Runtime

**Environment:**
- Python virtual environment (venv)
- Runtime: CPython

**Package Manager:**
- pip (pip-based)
- Lockfile: Not present (uses requirements.txt directly)

## Frameworks

**Core:**
- Ultralytics YOLO - Object detection framework (YOLO11/v26 variants)
- FastAPI - Web framework for REST API

**Data Processing & ML:**
- OpenCV (cv2) - Image processing
- PIL (Pillow) - Image loading and manipulation
- NumPy - Numerical operations
- Pandas - Data handling

**Visualization:**
- Matplotlib - Plotting and visualization
- Seaborn - Statistical visualization

**Utilities:**
- python-dotenv - Environment variable loading
- PyYAML - YAML file parsing (for data.yaml)
- tqdm - Progress bars

## Key Dependencies

**Critical:**
- ultralytics - YOLO model training and inference (latest version from pip)
- roboflow - Dataset download and management
- fastapi - REST API framework
- uvicorn[standard] - ASGI server for FastAPI

**Image Processing:**
- opencv-python - Computer vision operations
- pillow - Image file handling
- numpy - Array operations

**Data & Visualization:**
- pandas - Data manipulation
- matplotlib - Visualization
- seaborn - Statistical graphics

**Other:**
- python-multipart - File upload handling in FastAPI
- pyyaml - YAML configuration parsing
- tqdm - Training progress tracking

## Configuration

**Environment:**
- Environment variables loaded via python-dotenv from `.env` file
- Key variables: ROBOFLOW_API_KEY, ROBOFLOW_WORKSPACE, ROBOFLOW_PROJECT, ROBOFLOW_VERSION

**Model Configuration:**
- data.yaml - Dataset configuration (paths, class names)
- YOLO training parameters: epochs=25, imgsz=640

**API Configuration:**
- FastAPI app with title, description, version metadata
- Uvicorn runs on host=0.0.0.0, port=8000

## Platform Requirements

**Development:**
- Python 3.x
- Virtual environment (venv)
- pip for dependency management

**Production:**
- FastAPI-compatible ASGI server (uvicorn)
- ONNX runtime for optimized inference (optional, via model export)
- Model files: .pt (PyTorch) or .onnx (exported)

---

*Stack analysis: 2026-03-04*
