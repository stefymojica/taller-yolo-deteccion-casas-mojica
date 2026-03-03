import os
import glob
import cv2
from ultralytics import YOLO

def find_best_model():
    """Busca el mejor archivo .pt disponible en el proyecto."""
    # 1. Buscar en entrenamientos locales recientes
    local_runs = glob.glob("runs/detect/train*/weights/best.pt")
    local_runs.sort(key=os.path.getmtime, reverse=True)
    
    if local_runs:
        return local_runs[0]
    
    # 2. Buscar el de Colab en la carpeta models
    colab_model = "models/best_colab.pt"
    if os.path.exists(colab_model):
        return colab_model
    
    # 3. Fallback al modelo base si no hay nada entrenado
    return "yolo26n.pt"

def run_inference(image_path, model_path=None):
    """Script para inferir sobre imágenes nuevas usando el modelo .pt."""
    
    # Si no se pasa ruta, buscar automáticamente el mejor .pt
    if model_path is None:
        model_path = find_best_model()

    print(f"--- Cargando modelo: {model_path} ---")
    
    # Cargar el modelo YOLO
    model = YOLO(model_path)

    # Realizar la detección
    # Bajamos el umbral de confianza (conf=0.1) para ver detecciones iniciales
    results = model(image_path, conf=0.10, verbose=True)

    # Mostrar o guardar los resultados
    for i, result in enumerate(results):
        print(f"--- Detecciones encontradas: {len(result.boxes)} ---")
        result.show()  # Muestra la imagen interactiva
        
        # Guardar el resultado en la carpeta raíz
        result.save(filename=f'resultado_prueba_{i}.jpg')
        print(f"Imagen guardada como: resultado_prueba_{i}.jpg")

if __name__ == "__main__":
    # Ruta de una imagen de prueba del dataset
    # (Asegúrate de haber corrido python src/download_dataset.py primero)
    test_image = glob.glob("dataset/valid/images/*.jpg")
    
    if test_image:
        print(f"--- Probando detección con: {test_image[0]} ---")
        run_inference(test_image[0])
    else:
        print("Error: No se encontró imagen de prueba en dataset/valid/images/")
        print("Por favor, ejecuta 'python src/download_dataset.py' primero.")
