from ultralytics import YOLO

def train():
    model = YOLO("yolo26m.pt")

    results = model.train(
        data="dataset/data.yaml",
        epochs=25,
        imgsz=640,
        name="train_casas"
    )
    print("Entrenamiento finalizado.")

if __name__ == "__main__":
    train()
