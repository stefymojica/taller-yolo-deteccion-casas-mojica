# Taller de Detección de Casas con YOLO26

Este repositorio contiene los scripts y la configuración necesaria para entrenar y desplegar un modelo **YOLO26** enfocado en la detección de casas.

## Estructura del Proyecto

```text
taller-yolo-casas/
├── src/
│   ├── download_dataset.py # Descarga datos de Roboflow
│   ├── train_yolo.py       # Script de entrenamiento local
│   ├── export_model.py     # Convierte .pt (PyTorch) a .onnx (Producción)
│   ├── inferencia.py       # Script para probar detecciones
│   └── utils.py           # Utilidades extra
├── models/                 # Modelos finales (.onnx) y backups (.pt)
├── requirements.txt
├── .env                  # Variables de entorno (API Keys de Roboflow)
├── data.yaml              # Descriptor del dataset para YOLO
└── README.md
```

## Guía de Uso (Paso a Paso)

### 1. Configuración Inicial
Primero, instala las dependencias necesarias en tu ambiente virtual:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configurar Credenciales
Crea un archivo `.env` en la raíz (si no existe) y agrega tus claves de Roboflow:
```text
ROBOFLOW_API_KEY=tu_api_key_aqui
ROBOFLOW_WORKSPACE=tu_workspace
ROBOFLOW_PROJECT=tu_proyecto
ROBOFLOW_VERSION=2
```

### 3. Descargar el Dataset
Ejecuta el script para bajar las imágenes y etiquetas desde Roboflow:
```bash
python src/download_dataset.py
```

### 4. Entrenamiento
Tienes dos opciones para el entrenamiento:
*   **Local:** Ejecuta `python src/train_yolo.py`. Los resultados quedarán en la carpeta `runs/`.
*   **Google Colab:** Entrena en la nube y descarga el archivo `best.pt`. Guárdalo en `models/best_colab.pt`.

### 5. Exportar para Producción (Alta Velocidad)
Para que el modelo corra rápido en tu CPU o en una API, conviértelo al formato ONNX:
```bash
python src/export_model.py
```
*Este script priorizará el entrenamiento local más reciente y usará el de Colab como respaldo.*

### 6. Probar Inferencia
Finalmente, prueba que el modelo esté detectando casas correctamente:
```bash
python src/inferencia.py
```

---

> **Nota sobre Git:** Los archivos `.env`, la carpeta `dataset/`, la carpeta `runs/` y los modelos `.pt` están configurados en el `.gitignore` para no ser subidos al repositorio por seguridad y eficiencia.
