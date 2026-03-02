import os
from roboflow import Roboflow
from dotenv import load_dotenv

# Cargar variables desde el archivo .env
load_dotenv()

def download():
    """Descarga el dataset desde Roboflow usando variables de entorno."""
    api_key = os.getenv("ROBOFLOW_API_KEY")
    workspace = os.getenv("ROBOFLOW_WORKSPACE")
    project_name = os.getenv("ROBOFLOW_PROJECT")
    version_num = int(os.getenv("ROBOFLOW_VERSION", 1))

    if not api_key:
        print("Error: ROBOFLOW_API_KEY no encontrada en el archivo .env")
        return

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_name)
    version = project.version(version_num)
    
    # Descargar el dataset en una carpeta llamada 'dataset'
    print(f"Iniciando descarga de la versión {version_num}...")
    dataset = version.download("yolo26", location="./dataset")
    print(f"¡Hecho! Dataset descargado en: {dataset.location}")

if __name__ == "__main__":
    download()
