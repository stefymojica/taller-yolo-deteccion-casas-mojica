import os
import glob
from ultralytics import YOLO


def validate():
    def find_best_model():
        local_runs = glob.glob("runs/detect/train*/weights/best.pt")
        local_runs.sort(key=os.path.getmtime, reverse=True)

        if local_runs:
            print(f"Usando mejor modelo local: {local_runs[0]}")
            return local_runs[0]

        colab_model = "models/best_colab.pt"
        if os.path.exists(colab_model):
            print(f"Usando modelo de Colab: {colab_model}")
            return colab_model

        print("No se encontró un modelo entrenado. Usando modelo base yolov8m.pt")
        return "yolov8m.pt"

    model_path = find_best_model()
    model = YOLO(model_path)

    print(f"\n--- Iniciando validación con el modelo: {model_path} ---")

    # Evaluar sobre el set de validación (como en el notebook V1)
    metrics = model.val(
        data="dataset/data.yaml",
        imgsz=640,
        conf=0.25,
        iou=0.5,
        split="val",
    )

    # --- Métricas por clase ---
    print("\n📊 MÉTRICAS POR CLASE")
    print("=" * 50)
    names = model.names
    for i, name in names.items():
        print(f"\n🔹 Clase: {name}")
        print(f"   Precision : {metrics.box.p[i]:.3f}")
        print(f"   Recall    : {metrics.box.r[i]:.3f}")
        print(f"   mAP50     : {metrics.box.ap50[i]:.3f}")
        print(f"   mAP50-95  : {metrics.box.ap[i]:.3f}")

    # --- Métricas globales ---
    print("\n📊 MÉTRICAS GLOBALES")
    print("=" * 50)
    print(f"   mAP50     : {metrics.box.map50:.3f}")
    print(f"   mAP50-95  : {metrics.box.map:.3f}")

    print(f"\nCurvas y resultados guardados en: {metrics.save_dir}")
    print("Archivos generados: PR curve, F1 curve, confusion matrix, etc.")


if __name__ == "__main__":
    validate()
