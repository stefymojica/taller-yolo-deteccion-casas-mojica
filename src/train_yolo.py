from ultralytics import YOLO

def train():
    """Script de entrenamiento para YOLOv8."""
    # Cargar un modelo YOLO26m (medium) pre-entrenado
    # La versión 'm' es más inteligente que la 'n' (nano)
    model = YOLO("yolo26m.pt")

    # Entrenar el modelo
    # El archivo data.yaml define las rutas a las imágenes y etiquetas
    results = model.train(
        data="dataset/data.yaml",
        epochs=25,
        imgsz=640,
        name="train_casas"
    )
    print("Entrenamiento finalizado.")

if __name__ == "__main__":
    train()
