"""
Test script to verify ONNX model loads and runs inference correctly.
"""

import onnxruntime as ort
import numpy as np
from PIL import Image
import os

# Paths
MODEL_PATH = "../models/house_detector_prod.onnx"
TEST_IMAGE = "../dataset/train/images/14417850313_aed39942bf_b_jpg.rf.8bc672fbf52b6a751e9886cabcb1c249.jpg"

print("=" * 50)
print("Testing ONNX Model")
print("=" * 50)

# 1. Load ONNX model
print("\n1. Loading ONNX model...")
try:
    session = ort.InferenceSession(MODEL_PATH)
    print(f"   Model loaded successfully!")

    # Get input name
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    print(f"   Input name: {input_name}")
    print(f"   Input shape: {input_shape}")

    # Get output names
    output_names = [o.name for o in session.get_outputs()]
    print(f"   Output names: {output_names}")

except Exception as e:
    print(f"   ERROR loading model: {e}")
    exit(1)

# 2. Load test image
print("\n2. Loading test image...")
try:
    image = Image.open(TEST_IMAGE).convert("RGB")
    original_size = image.size
    print(f"   Image loaded: {original_size}")

    # Resize to 640x640 (model input size)
    image_resized = image.resize((640, 640))
    img_array = np.array(image_resized).astype(np.float32) / 255.0

    # Transpose from HWC to CHW
    img_transposed = np.transpose(img_array, (2, 0, 1))
    # Add batch dimension
    img_batch = np.expand_dims(img_transposed, axis=0)

    print(f"   Image processed: shape={img_batch.shape}")

except Exception as e:
    print(f"   ERROR loading image: {e}")
    exit(1)

# 3. Run inference
print("\n3. Running inference...")
try:
    outputs = session.run(output_names, {input_name: img_batch})

    # Parse outputs (YOLO format: [batch, num_boxes, num_props])
    # Output 0: predictions (1, 84, 8400) for yolov8
    # Output 1: prototype if using maskrcnn

    print(f"   Number of outputs: {len(outputs)}")
    for i, out in enumerate(outputs):
        print(f"   Output {i} shape: {out.shape}")
        print(f"   Output {i} dtype: {out.dtype}")

    # For YOLO ONNX, typically output shape is (1, 84, 8400)
    # 84 = 4 (bbox) + 80 (classes)
    predictions = outputs[0]
    print(f"\n   Predictions shape: {predictions.shape}")

    # Transpose to (1, 8400, 84)
    predictions_transposed = np.transpose(predictions, (0, 2, 1))
    print(f"   Transposed shape: {predictions_transposed.shape}")

    # Extract boxes and scores
    boxes = predictions_transposed[0, :, :4]  # x, y, w, h
    scores = predictions_transposed[0, :, 4:]  # class scores

    # Get max score and class
    max_scores = np.max(scores, axis=1)
    max_classes = np.argmax(scores, axis=1)

    # Filter by confidence threshold
    conf_threshold = 0.25
    mask = max_scores > conf_threshold

    filtered_boxes = boxes[mask]
    filtered_scores = max_scores[mask]
    filtered_classes = max_classes[mask]

    print(f"\n4. Results:")
    print(f"   Total detections: {np.sum(mask)}")
    print(f"   Boxes shape: {filtered_boxes.shape}")
    print(
        f"   Confidence range: {filtered_scores.min():.3f} - {filtered_scores.max():.3f}"
    )

    if len(filtered_boxes) > 0:
        print(f"\n   Top 5 detections:")
        top_indices = np.argsort(filtered_scores)[::-1][:5]
        for idx in top_indices:
            print(
                f"     - Class: {filtered_classes[idx]}, Conf: {filtered_scores[idx]:.3f}, Box: {filtered_boxes[idx]}"
            )

    print("\n" + "=" * 50)
    print("SUCCESS: ONNX model works correctly!")
    print("=" * 50)

except Exception as e:
    print(f"   ERROR during inference: {e}")
    import traceback

    traceback.print_exc()
    exit(1)
