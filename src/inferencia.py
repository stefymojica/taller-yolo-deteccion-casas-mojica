import os
import glob
import cv2
from ultralytics import YOLO

def find_best_model():
    local_runs = glob.glob("runs/detect/train*/weights/best.pt")
    local_runs.sort(key=os.path.getmtime, reverse=True)
    
    if local_runs:
        return local_runs[0]
    
    colab_model = "models/best_colab.pt"
    if os.path.exists(colab_model):
        return colab_model
    
    return "yolo26n.pt"

def run_inference(image_path, model_path=None):
    if model_path is None:
        model_path = find_best_model()

    print(f"--- Cargando modelo: {model_path} ---")
    model = YOLO(model_path)

    results = model(image_path, conf=0.10, verbose=True)

    for i, result in enumerate(results):
        print(f"--- Detecciones encontradas: {len(result.boxes)} ---")
        result.show()
        result.save(filename=f'resultado_prueba_{i}.jpg')
        print(f"Imagen guardada como: resultado_prueba_{i}.jpg")

if __name__ == "__main__":
    test_image = glob.glob("dataset/valid/images/*.jpg")
    
    if test_image:
        print(f"--- Probando detección con: {test_image[0]} ---")
        run_inference(test_image[0])
    else:
        print("Error: No se encontró imagen de prueba en dataset/valid/images/")
        print("Por favor, ejecuta 'python src/download_dataset.py' primero.")
