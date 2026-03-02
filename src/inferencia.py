from ultralytics import YOLO
import cv2

def run_inference(image_path, model_path="models/house_detector_prod.onnx"):
    """Script para inferir sobre imágenes nuevas con el modelo de producción (ONNX)."""
    # Cargar el modelo YOLO
    model = YOLO(model_path)

    # Realizar la detección
    # Bajamos el umbral de confianza (conf=0.1) para ver si está detectando algo
    results = model(image_path, conf=0.10, verbose=True)

    # Mostrar o guardar los resultados
    for i, result in enumerate(results):
        print(f"--- Detecciones encontradas: {len(result.boxes)} ---")
        result.show()  # Muestra la imagen con las cajas de detección
        # Guardar el resultado en la carpeta raíz para revisión manual
        result.save(filename=f'resultado_prueba_{i}.jpg')
        print(f"Imagen guardada como: resultado_prueba_{i}.jpg")

if __name__ == "__main__":
    # Ruta de una imagen de prueba del dataset de validación
    test_image = "dataset/valid/images/73985309-san-cristobal-de-las-casas-chiapas-mexico-february-19-2017-san-cristobal-de-las-casas-is-known_jpg.rf.ed6462c6ef77f6405b7094441ad4a719.jpg"
    
    print(f"--- Probando detección con: {test_image} ---")
    run_inference(test_image)
    print("Prueba completada.")
