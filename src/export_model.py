import os
import glob
import shutil
from ultralytics import YOLO

def export():
    if not os.path.exists("models"):
        os.makedirs("models")
        print("Carpeta 'models/' creada.")

    local_runs = glob.glob("runs/detect/train*/weights/best.pt")
    local_runs.sort(key=os.path.getmtime, reverse=True)
    
    potential_paths = []
    if local_runs:
        potential_paths.append(local_runs[0])
    
    potential_paths.append("models/best_colab.pt")
    
    weights_path = None
    for path in potential_paths:
        if os.path.exists(path):
            weights_path = path
            break

    if not weights_path:
        print("Error: No se encontró ningún archivo 'best.pt' para exportar.")
        print("Asegúrate de haber entrenado localmente o tener 'models/best_colab.pt'.")
        return

    print(f"--- Usando modelo para exportar: {weights_path} ---")

    print(f"Cargando modelo desde {weights_path}...")
    model = YOLO(weights_path)

    print("Exportando a ONNX (opset=17)...")
    onnx_path = model.export(format="onnx", imgsz=640, dynamic=True, opset=17)
    
    target_path = "models/house_detector_prod.onnx"
    shutil.copy(onnx_path, target_path)
    
    print("-" * 30)
    print(f"¡Exportación exitosa!")
    print(f"Origen: {weights_path}")
    print(f"Original (.pt): {os.path.getsize(weights_path) / (1024*1024):.2f} MB")
    print(f"Producción (.onnx): {os.path.getsize(target_path) / (1024*1024):.2f} MB")
    print(f"Modelo guardado en: {target_path}")
    print("-" * 30)

if __name__ == "__main__":
    export()
