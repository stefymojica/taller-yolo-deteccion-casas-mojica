# Coding Conventions

**Analysis Date:** 2026-03-04

## Language & Framework

**Primary Language:** Python 3.x

**Key Frameworks:**
- **YOLO** (Ultralytics) - Object detection model training and inference
- **FastAPI** - REST API for model deployment
- **Roboflow** - Dataset management

## Naming Patterns

**Files:**
- Snake case, descriptive: `train_yolo.py`, `val_metrics.py`, `download_dataset.py`
- Action-based naming: `{action}_{subject}.py` pattern

**Functions:**
- Snake case, action-oriented: `download()`, `train()`, `export()`, `find_best_model()`
- Verb-first naming: `{verb}_{object}()` pattern
- Examples in `src/utils.py`:
  - `convert_format()`
  - `visualize_dataset()`

**Variables:**
- Snake case throughout codebase
- Descriptive, lowercase: `api_key`, `model_path`, `weights_path`, `map50`, `version_num`
- No single-letter variables (except in comprehensions)

**Constants:**
- Uppercase for string literals: `"yolo26m.pt"`, `"runs/detect/train*/weights/best.pt"`
- Path strings use forward slashes

## Code Style

**Formatting:**
- No explicit formatter configured (no `ruff`, `black`, or `pyproject.toml` formatting rules)
- Standard Python indentation (4 spaces)
- Line length: Not enforced (no tools configured)

**Linting:**
- No linting tools configured (no `ruff`, `pylint`, `flake8`)
- `.gitignore` includes generic Python linting cache directories: `.pyre/`, `.pytype/`, `.mypy_cache/`

**Line Structure:**
- Maximum ~50-70 lines per file in source scripts
- Functions typically 15-30 lines
- Clear separation between imports, function definitions, and main execution blocks

## Import Organization

**Order:**
1. Standard library: `os`, `glob`, `io`, `cv2`, `numpy`
2. Third-party packages: `roboflow`, `ultralytics`, `fastapi`, `PIL`
3. Local imports: None in current codebase

**Path Style:**
- No path aliases configured (no `sys.path` manipulation)
- Relative imports from same package: Not used

**Examples from `src/main_api.py`:**
```python
import io
import os
import glob
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
from PIL import Image
```

## Error Handling

**Patterns:**
- **Print-based errors:** Most scripts use `print()` for error messages
  ```python
  # Example from src/download_dataset.py
  if not api_key:
      print("Error: ROBOFLOW_API_KEY no encontrada en el archivo .env")
      return
  ```
  
- **API errors:** HTTPException for FastAPI endpoints
  ```python
  # Example from src/main_api.py
  if not file.content_type.startswith("image/"):
      raise HTTPException(status_code=400, detail="El archivo enviado no es una imagen.")
  ```

- **Try-except:** Minimal usage, only in API endpoint
  ```python
  try:
      contents = await file.read()
      image = Image.open(io.BytesIO(contents)).convert("RGB")
      # ...
  except Exception as e:
      raise HTTPException(status_code=500, detail=f"Error procesando la imagen: {str(e)}")
  ```

**No formal logging:** The codebase uses `print()` statements instead of Python's `logging` module.

## Logging

**Framework:** None - uses `print()` exclusively

**Patterns:**
- Status messages: `print(f"--- Iniciando descarga de la versión {version_num}...")`
- Success messages: `print(f"¡Hecho! Dataset descargado en: {dataset.location}")`
- Error messages: `print("Error: ...")`
- Separator lines: `print("-" * 30)` or `print("="*40)`

**When to Log:**
- Before starting operations
- After completing operations
- On errors
- To display metrics/results

## Comments

**When to Comment:**
- Minimal inline comments in current codebase
- Occasional explanatory comments for non-obvious behavior:
  ```python
  # Example from src/train_yolo.py
  # En YOLOv8/v11, results.results_dict contiene las métricas finales
  map50 = results.results_dict.get('metrics/mAP50(B)', 0)
  ```

**Spanish Language:** Comments and print messages are in Spanish

**No Docstrings:** Functions generally lack docstrings

## Function Design

**Size:** Small, focused functions (10-30 lines typical)

**Parameters:** 
- Simple parameters with defaults
- Example: `def run_inference(image_path, model_path=None):`

**Return Values:**
- Implicit returns (no return statement = None)
- Direct printing of results rather than returning values

**Structure Pattern:**
```python
def function_name():
    # Setup/loading
    # Processing
    # Output/results
    
if __name__ == "__main__":
    function_name()
```

## Module Design

**Exports:** Not applicable - scripts are run directly, not imported as modules

**Entry Points:** 
- `if __name__ == "__main__":` block in each script
- Calls main function defined in same file

**No Package Structure:**
- Flat `src/` directory with individual scripts
- No `__init__.py` files
- No subdirectories for modules

## Type Hints

**Usage:** Not detected in current codebase
- No type annotations on functions or variables
- Example: `def download():` instead of `def download() -> None:`

## Configuration

**Environment Variables:**
- Loaded via `python-dotenv` in `download_dataset.py`: `load_dotenv()`
- Variables accessed via `os.getenv()`
- Spanish error messages for missing vars

**No Config Files:**
- No `pyproject.toml` for project configuration
- No `setup.py` or `setup.cfg`
- No `tox.ini` or `noxfile.py`

---

*Convention analysis: 2026-03-04*
