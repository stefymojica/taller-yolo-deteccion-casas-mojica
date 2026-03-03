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

## 🚀 Comandos Rápidos

Una vez configurado el ambiente y el `.env`, estos son los comandos principales:

| Paso | Comando | Descripción |
| :--- | :--- | :--- |
| **1. Descargar** | `python src/download_dataset.py` | Baja el dataset de Roboflow a `/dataset` |
| **2. Entrenar** | `python src/train_yolo.py` | Inicia entrenamiento local (YOLO26 Medium) |
| **3. Exportar** | `python src/export_model.py` | Convierte el mejor `best.pt` a `house_detector_prod.onnx` |
| **4. Inferir** | `python src/inferencia.py` | Prueba el modelo con imágenes de validación |

---

## Guía de Uso Detallada

### 1. Configuración Inicial
Instala las dependencias en tu ambiente virtual:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configurar Credenciales
Crea un archivo `.env` en la raíz con tus claves de Roboflow:
```text
ROBOFLOW_API_KEY=tu_api_key_aqui
ROBOFLOW_WORKSPACE=tu_workspace
ROBOFLOW_PROJECT=tu_proyecto
ROBOFLOW_VERSION=2
```

### 3. Descarga de Datos
Para obtener las imágenes etiquetadas:
```bash
python src/download_dataset.py
```

### 4. Entrenamiento del Modelo
Puedes entrenar localmente o en la nube:
*   **Local:** `python src/train_yolo.py`. Los resultados se guardan en `runs/detect/train_casasX`.
*   **Google Colab:** Si entrenas en Colab, descarga el `best.pt` y colócalo en `models/best_colab.pt`.

### 5. Exportación a Producción (Optativo pero Recomendado)
Para optimizar el modelo para CPUs o servidores:
```bash
python src/export_model.py
```
*El script detectará automáticamente el entrenamiento más reciente o el archivo de Colab.*

### 6. Ejecución de Inferencia
Para validar los resultados visualmente:
```bash
python src/inferencia.py
```

---

> **Nota sobre Git:** Los archivos `.env`, la carpeta `dataset/`, la carpeta `runs/` y los modelos `.pt` están configurados en el `.gitignore` para no ser subidos al repositorio por seguridad y eficiencia.
