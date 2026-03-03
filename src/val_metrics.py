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
    
    results = model.val(data="dataset/data.yaml")

    map50 = results.results_dict.get('metrics/mAP50(B)', 0)
    precision = results.results_dict.get('metrics/precision(B)', 0)
    recall = results.results_dict.get('metrics/recall(B)', 0)

    print("\n" + "="*40)
    print("      REPORTA DE MÉTRICAS (VALIDACIÓN)")
    print("="*40)
    print(f"Precision:  {precision:.4f}")
    print(f"Recall:     {recall:.4f}")
    print(f"mAP@0.5:    {map50:.4f}")
    print("="*40)

if __name__ == "__main__":
    validate()
