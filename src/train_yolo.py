from ultralytics import YOLO


def train():
    # YOLOv8m - modelo medium con mejor precisión que nano
    model = YOLO("yolov8m.pt")

    results = model.train(
        data="dataset/data.yaml",
        epochs=100,  # TRAIN-03: 100 epochs
        imgsz=1280,  # TRAIN-04: resolución 1280px
        optimizer="SGD",  # TRAIN-05: optimizer SGD
        momentum=0.937,  # TRAIN-05: momentum para SGD
        name="train_casas",
        # TRAIN-03: Early stopping con patience=20
        patience=20,
        # TRAIN-01: Data augmentation habilitada
        hsv_h=0.015,  # Hue augmentation
        hsv_s=0.7,  # Saturation augmentation
        hsv_v=0.4,  # Value augmentation
        degrees=0.0,  # Rotation
        translate=0.1,  # Translation
        scale=0.5,  # Scaling
        shear=0.0,  # Shearing
        perspective=0.0,  # Perspective
        flipud=0.0,  # Vertical flip
        fliplr=0.5,  # Horizontal flip
        mosaic=1.0,  # Mosaic augmentation
        mixup=0.0,  # Mixup augmentation
        #batch=16,    #para GPU
        #workers=4,   #para GPU
        #device=0,    #para GPU
        
    )

    # Reportar métricas finales
    print("\n" + "=" * 40)
    print("   MÉTRICAS FINALES DEL ENTRENAMIENTO")
    print("=" * 40)
    # En YOLOv8/v11, results.results_dict contiene las métricas finales
    map50 = results.results_dict.get("metrics/mAP50(B)", 0)
    precision = results.results_dict.get("metrics/precision(B)", 0)
    recall = results.results_dict.get("metrics/recall(B)", 0)

    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"mAP@0.5:   {map50:.4f}")
    print("=" * 40)
    print("Entrenamiento finalizado.")


if __name__ == "__main__":
    train()
