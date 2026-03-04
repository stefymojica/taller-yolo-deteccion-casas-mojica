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

        print("No se encontró un modelo entrenado. Usando modelo base yolo26m.pt")
        return "yolo26m.pt"

    model_path = find_best_model()
    model = YOLO(model_path)

    print(f"\n--- Iniciando validación con el modelo: {model_path} ---")
    print("--- Evaluando en TEST SET ---")

    # EVAL-01: Evaluar en test set separado
    results = model.val(data="dataset/data.yaml", split="test")

    # EVAL-02: Extraer métricas mAP50-95 y mAP50
    map50_95 = results.results_dict.get("metrics/mAP50-95(B)", 0)
    map50 = results.results_dict.get("metrics/mAP50(B)", 0)
    precision = results.results_dict.get("metrics/precision(B)", 0)
    recall = results.results_dict.get("metrics/recall(B)", 0)

    # EVAL-04: Calcular F1-score para clase "casa"
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    print("\n" + "=" * 50)
    print("      REPORTE DE MÉTRICAS (TEST SET)")
    print("=" * 50)
    print(f"Precision:       {precision:.4f}")
    print(f"Recall:          {recall:.4f}")
    print(f"mAP@0.5:         {map50:.4f}")
    print(f"mAP@0.5:0.95:    {map50_95:.4f}")
    print(f"F1-Score:        {f1:.4f}")
    print("=" * 50)
    print("Clase: casa")
    print("-" * 50)

    # EVAL-03: Curvas generadas automáticamente por Ultralytics
    print(f"\nCurvas y resultados guardados en: runs/detect/val12/")
    print("Archivos generados: PR curve, F1 curve, confusion matrix, etc.")


if __name__ == "__main__":
    validate()
