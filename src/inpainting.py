import os
import sys
import glob
import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO


def find_best_model():
    """Busca el mejor modelo entrenado disponible."""
    local_runs = glob.glob("runs/detect/train*/weights/best.pt")
    local_runs.sort(key=os.path.getmtime, reverse=True)

    if local_runs:
        return local_runs[0]

    colab_model = "models/best_colab.pt"
    if os.path.exists(colab_model):
        return colab_model

    return "yolov8m.pt"


def load_lama_model():
    """Carga el modelo LaMa forzando CPU (compatible con Mac sin CUDA)."""
    from simple_lama_inpainting.utils import download_model

    lama_url = "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt"
    model_path = download_model(lama_url)

    # Forzar carga en CPU (el modelo fue guardado con CUDA)
    model = torch.jit.load(model_path, map_location="cpu")
    model.eval()
    return model


def remove_postes(image_path, model_path=None, output_dir="resultados"):
    """
    Detecta postes en una imagen y los elimina usando inpainting (LaMa).

    Pasos:
      1. Cargar modelo YOLO entrenado
      2. Detectar objetos en la imagen
      3. Crear máscara binaria solo con las detecciones de clase "poste"
      4. Dilatar la máscara para cubrir bordes
      5. Aplicar inpainting con LaMa
      6. Guardar resultado
    """
    if model_path is None:
        model_path = find_best_model()

    print(f"--- Cargando modelo: {model_path} ---")
    model = YOLO(model_path)

    # 1. Cargar imagen
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: No se pudo cargar la imagen: {image_path}")
        return

    print(f"--- Procesando imagen: {image_path} ---")

    # 2. Detectar objetos
    results = model(image)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()

    # 3. Crear máscara solo con postes
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    postes_encontrados = 0

    for box, cls in zip(boxes, classes):
        if model.names[int(cls)] == "poste":
            x1, y1, x2, y2 = map(int, box)
            mask[y1:y2, x1:x2] = 255
            postes_encontrados += 1

    if postes_encontrados == 0:
        print("No se detectaron postes en la imagen.")
        return

    print(f"--- Postes detectados: {postes_encontrados} ---")

    # 4. Dilatar máscara para cubrir bordes
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # 5. Aplicar inpainting con LaMa (forzando CPU)
    print("--- Aplicando inpainting (LaMa)... ---")
    lama_model = load_lama_model()

    # Preparar imagen y máscara como tensores
    from simple_lama_inpainting.utils import prepare_img_and_mask
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    mask_pil = Image.fromarray(mask)
    device = torch.device("cpu")
    img_tensor, mask_tensor = prepare_img_and_mask(image_pil, mask_pil, device)

    with torch.inference_mode():
        inpainted = lama_model(img_tensor, mask_tensor)
        result_np = inpainted[0].permute(1, 2, 0).detach().cpu().numpy()
        result_np = np.clip(result_np * 255, 0, 255).astype(np.uint8)
        # Convertir de RGB a BGR para guardar con OpenCV
        result_np = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)

    # 6. Guardar resultado
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)
    output_path = os.path.join(output_dir, f"{name}_sin_postes{ext}")
    cv2.imwrite(output_path, result_np)

    print(f"--- Resultado guardado en: {output_path} ---")
    print("Inpainting finalizado.")


if __name__ == "__main__":
    # Si se pasa una ruta como argumento, usarla; si no, buscar en validación
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        test_images = glob.glob("dataset/valid/images/*.jpg")
        if test_images:
            image_path = test_images[0]
            print(f"--- Usando imagen de validación: {image_path} ---")
        else:
            print("Error: No se encontró imagen de prueba.")
            print("Uso: python src/inpainting.py <ruta_imagen>")
            print("  o ejecuta 'python src/download_dataset.py' para descargar el dataset.")
            sys.exit(1)

    remove_postes(image_path)
