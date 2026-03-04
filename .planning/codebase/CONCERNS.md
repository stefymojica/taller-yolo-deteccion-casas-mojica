# Codebase Concerns

**Analysis Date:** 2026-03-04

## Tech Debt

### Duplicate Model Finding Logic
- **Issue:** The `find_best_model()` function is duplicated in 4 files with identical implementation
- **Files:** `src/inferencia.py`, `src/main_api.py`, `src/val_metrics.py`, `src/export_model.py`
- **Impact:** Code maintenance burden; inconsistent behavior if one is updated
- **Fix approach:** Create a shared utility module in `src/utils.py` with a single `find_best_model()` function

### Empty Utility Functions
- **Issue:** `src/utils.py` contains empty stub functions with no implementation
- **Files:** `src/utils.py` (lines 1-5)
- **Functions:** `convert_format()`, `visualize_dataset()`
- **Impact:** Dead code; confusion for developers
- **Fix approach:** Either implement these functions or remove them from the codebase

### Hardcoded Configuration Values
- **Issue:** Configuration values are hardcoded throughout the codebase instead of centralized
- **Locations:**
  - `src/train_yolo.py`: epochs=25, imgsz=640
  - `src/inferencia.py`: conf=0.10
  - `src/main_api.py`: conf=0.25, imgsz=640
- **Impact:** Difficult to change configuration; inconsistent settings across scripts
- **Fix approach:** Create a `config.py` or `config.yaml` with all configurable parameters

## Known Issues

### Model Loading at Module Import Time
- **Issue:** In `src/main_api.py`, the model is loaded at module level (lines 30-32) before any request handling
- **Files:** `src/main_api.py`
- **Symptom:** API fails to start if no model is found; no graceful degradation
- **Trigger:** Running `python src/main_api.py` without a trained model
- **Workaround:** Ensure a valid model exists before starting the API

### Inconsistent Model Base Names
- **Issue:** Different scripts default to different base models when no trained model is found
- **Files:** `src/train_yolo.py` uses "yolo26m.pt", `src/inferencia.py` and others default to "yolo26n.pt" or "yolo26m.pt" inconsistently
- **Impact:** Unexpected behavior when running without a trained model

### Data.yaml Not Found in Dataset
- **Issue:** The training script references `dataset/data.yaml` but the file wasn't found in the expected location
- **Files:** `src/train_yolo.py`, `src/val_metrics.py`
- **Impact:** Training/validation will fail if data.yaml is not properly downloaded
- **Workaround:** Run `python src/download_dataset.py` before training

## Security Considerations

### API Lacks Authentication
- **Issue:** The FastAPI endpoint `/predict` has no authentication or rate limiting
- **Files:** `src/main_api.py`
- **Risk:** Anyone can upload images and use compute resources; potential for abuse
- **Recommendations:** Add API key authentication, rate limiting, and request size limits

### No Input Validation for Image Files
- **Issue:** API only checks content-type prefix but doesn't validate actual image format
- **Files:** `src/main_api.py` (line 36-37)
- **Risk:** Malformed images could cause unexpected behavior or crashes
- **Recommendations:** Add proper image format validation and size limits

### Sensitive Data in .env Not Protected
- **Issue:** `.env` file contains Roboflow API credentials but standard .gitignore may not prevent all accidental commits
- **Files:** `.env`, `.gitignore`
- **Risk:** API key exposure if committed to version control
- **Current mitigation:** `.env` is in `.gitignore` (line 80)
- **Recommendations:** Add pre-commit hooks to verify no secrets are staged

## Performance Bottlenecks

### No Async Model Loading
- **Issue:** Model loading in API is synchronous, blocking startup
- **Files:** `src/main_api.py` (lines 30-32)
- **Impact:** Slow API startup; cannot serve requests until model fully loaded
- **Improvement path:** Use lazy loading or background model loading

### Image Processing Without Batch Optimization
- **Issue:** Each image is processed individually with no batching
- **Files:** `src/main_api.py`
- **Impact:** Lower throughput for multiple simultaneous requests
- **Improvement path:** Implement request queuing with batch processing

### No Model Caching in Inference
- **Issue:** Model is reloaded on each script invocation in some cases
- **Files:** `src/inferencia.py`, `src/val_metrics.py`
- **Impact:** Slower inference cycles
- **Improvement path:** Implement singleton pattern or caching for model instances

## Fragile Areas

### Roboflow Download Without Error Handling
- **Issue:** `src/download_dataset.py` lacks robust error handling for network failures
- **Files:** `src/download_dataset.py`
- **Why fragile:** Network timeouts or invalid credentials cause unclear errors
- **Safe modification:** Add retry logic, better error messages, and validation of environment variables

### Validation Script Depends on Training Output
- **Issue:** Validation requires the dataset to be in a specific structure from Roboflow
- **Files:** `src/val_metrics.py`
- **Why fragile:** Breaks if dataset is moved or corrupted
- **Safe modification:** Add dataset structure validation before running

### ONNX Export Hardcoded Parameters
- **Issue:** Export script uses fixed opset version and image size
- **Files:** `src/export_model.py` (line 37)
- **Why fragile:** May not be optimal for all deployment targets
- **Safe modification:** Make export parameters configurable

## Scaling Limits

### No GPU Memory Management
- **Current capacity:** Defaults to auto-detect GPU; no explicit memory limits
- **Limit:** May crash on systems with limited VRAM
- **Scaling path:** Add device selection and memory management options

### No Horizontal Scaling Support
- **Current capacity:** Single API instance
- **Limit:** Cannot handle high concurrency
- **Scaling path:** Add stateless API design for container orchestration

### Dataset Size Constraints
- **Current capacity:** Unknown; depends on Roboflow project
- **Limit:** No data augmentation or streaming for large datasets
- **Scaling path:** Implement streaming data loading for larger datasets

## Dependencies at Risk

### Ultralytics Version Pinning
- **Issue:** No version pin in requirements.txt; uses "ultralytics" without version constraint
- **Files:** `requirements.txt` (line 1)
- **Risk:** Breaking changes in future releases could break training/inference
- **Migration plan:** Pin to specific version (e.g., `ultralytics==8.0.0`) after testing

### Roboflow SDK Dependency
- **Issue:** Tight coupling to Roboflow for dataset download
- **Files:** `src/download_dataset.py`
- **Risk:** Roboflow API changes could break dataset download
- **Migration plan:** Implement alternative dataset loading (local files, other sources)

## Missing Critical Features

### No Unit Tests
- **Problem:** No test suite exists for any module
- **Blocks:** Confidence in refactoring; reproducible validation of functionality

### No Model Version Tracking
- **Problem:** No clear way to determine which model version is deployed
- **Blocks:** Reproducibility; rollback capabilities

### No Logging Configuration
- **Problem:** Scripts use print() statements; no structured logging
- **Blocks:** Production debugging; performance monitoring

### No Data Augmentation Pipeline
- **Problem:** Training uses default YOLO augmentations only
- **Blocks:** Performance improvement through custom augmentation

## Test Coverage Gaps

### No Test Files
- **What's not tested:** All functionality
- **Files:** Entire codebase lacks tests
- **Risk:** Bugs could go unnoticed; breaking changes not caught
- **Priority:** High

### No Integration Tests for API
- **What's not tested:** API endpoint behavior, error handling, response formats
- **Files:** `src/main_api.py`
- **Risk:** API could return incorrect responses without detection
- **Priority:** High

### No Model Validation Tests
- **What's not tested:** Model output correctness, edge cases
- **Files:** `src/inferencia.py`, `src/val_metrics.py`
- **Risk:** Model could produce incorrect detections without detection
- **Priority:** Medium

---

*Concerns audit: 2026-03-04*
