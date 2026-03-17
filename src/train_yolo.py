from ultralytics import YOLO


def train():
    # YOLOv8m - modelo medium con mejor precisión
    model = YOLO("yolov8m.pt")

    results = model.train(
        data="dataset/data.yaml",
        epochs=100,            # más épocas, con patience para parar solo
        imgsz=640,             # mejora detección de objetos delgados (postes)
        batch=16,
        patience=25,           # espera 25 épocas sin mejora antes de parar
        optimizer="AdamW",     # mejor que SGD para datasets pequeños
        lr0=0.001,             # learning rate inicial
        lrf=0.01,              # learning rate final (lr0 * lrf)
        weight_decay=0.0005,
        warmup_epochs=5.0,     # primeras 5 épocas suaves para estabilizar

        # --- Augmentaciones complementarias a las de Roboflow ---
        fliplr=0.0,            # apagar — Roboflow ya lo hizo
        flipud=0.0,            # postes no aparecen boca abajo
        hsv_h=0.01,            # tono muy suave
        hsv_s=0.3,             # saturación moderada
        hsv_v=0.3,             # brillo moderado
        degrees=3.0,           # rotación leve — útil para postes inclinados
        translate=0.1,         # desplazamiento leve
        scale=0.4,             # zoom — ayuda a detectar postes lejos y cerca
        shear=2.0,             # distorsión leve
        perspective=0.0001,    # simula ángulos de cámara distintos
        mosaic=0.5,            # combina imágenes — ayuda con pocas muestras
        mixup=0.0,             # apagar — puede confundir con solo 2 clases
        copy_paste=0.0,        # apagar — requiere segmentación

        # --- Configuración de pérdida ---
        cls=0.3,               # reducir peso de clasificación (solo 2 clases)
        box=7.5,               # aumentar peso de bounding box (localizar el palo)

        name="train_postes_v2",
        exist_ok=True,
    )

    # Reportar métricas finales
    print("\n" + "=" * 40)
    print("   MÉTRICAS FINALES DEL ENTRENAMIENTO")
    print("=" * 40)
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
