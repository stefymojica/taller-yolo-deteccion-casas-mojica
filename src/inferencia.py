from ultralytics import YOLO
import cv2

def run_inference(image_path, model_path="models/best.pt"):
    """Script para inferir sobre imágenes nuevas con YOLO26."""
    # Cargar el modelo YOLO
    model = YOLO(model_path)

    # Realizar la detección
    results = model(image_path)

    # Mostrar o guardar los resultados
    for result in results:
        result.show()  # Muestra la imagen con las cajas de detección
        # result.save(filename='resultado.jpg')  # Guarda la detección

if __name__ == "__main__":
    # Sustituir con la ruta real de la imagen
    # run_inference("path/to/image.jpg")
    print("Módulo de inferencia listo.")
