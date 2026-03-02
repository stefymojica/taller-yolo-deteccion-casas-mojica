from ultralytics import YOLO

def train():
    """Script de entrenamiento para YOLOv8."""
    # Cargar un modelo YOLOv8n (nano) pre-entrenado
    model = YOLO("yolov8n.pt")

    # Entrenar el modelo
    # El archivo data.yaml define las rutas a las imágenes y etiquetas
    results = model.train(
        data="data.yaml",
        epochs=50,
        imgsz=640,
        name="train_casas"
    )
    print("Entrenamiento finalizado.")

if __name__ == "__main__":
    train()
