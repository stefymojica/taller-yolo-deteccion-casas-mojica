# Taller de Detección de Casas con YOLO26

Este repositorio contiene los scripts y la configuración necesaria para entrenar un modelo YOLO26 enfocado en la detección de casas.

## Estructura del Proyecto

```text
taller-yolo-casas/
├── src/
│   ├── download_dataset.py # Descarga datos de Roboflow
│   ├── train_yolo.py       # Script de entrenamiento
│   ├── inferencia.py       # Script para inferencia
│   └── utils.py           # Utilidades
├── models/                 # Pesos guardados (.pt)
├── requirements.txt
├── .env                  # Variables de entorno (API Keys)
├── data.yaml              # Descriptor del dataset para YOLO
└── README.md
```

## Instrucciones

1. Instalar dependencias: `pip install -r requirements.txt`
2. Configurar `.env` con tu API Key de Roboflow.
3. Descargar el dataset: `python src/download_dataset.py`
4. Configurar `data.yaml` con las rutas correctas (si es necesario).
5. Entrenar: `python src/train_yolo.py`
6. Inferir: `python src/inferencia.py`

> **Nota sobre Git:** Los archivos `.env`, la carpeta `dataset/` y los resultados en `runs/` están configurados en el `.gitignore` para no ser subidos al repositorio por seguridad y eficiencia.
