from ultralytics import YOLO

def train():
    model = YOLO("yolo26m.pt")

    results = model.train(
        data="dataset/data.yaml",
        epochs=25,
        imgsz=640,
        name="train_casas"
    )

    # Reportar métricas finales
    print("\n" + "="*40)
    print("   MÉTRICAS FINALES DEL ENTRENAMIENTO")
    print("="*40)
    # En YOLOv8/v11, results.results_dict contiene las métricas finales
    map50 = results.results_dict.get('metrics/mAP50(B)', 0)
    precision = results.results_dict.get('metrics/precision(B)', 0)
    recall = results.results_dict.get('metrics/recall(B)', 0)
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"mAP@0.5:   {map50:.4f}")
    print("="*40)
    print("Entrenamiento finalizado.")

if __name__ == "__main__":
    train()
