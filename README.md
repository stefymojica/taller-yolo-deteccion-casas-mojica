# Taller de Detección de Casas con YOLO

Este repositorio contiene los scripts y la configuración necesaria para entrenar un modelo YOLO enfocado en la detección de casas.

## Estructura del Proyecto

```text
taller-yolo-casas/
├── src/
│   ├── train_yolo.py   # Script de entrenamiento
│   ├── inferencia.py   # Script para inferencia
│   └── utils.py       # Utilidades
├── models/             # Pesos guardados (.pt)
├── requirements.txt
├── data.yaml          # Descriptor del dataset para YOLO
└── README.md
```

## Instrucciones

1. Instalar dependencias: `pip install -r requirements.txt`
2. Configurar `data.yaml` con las rutas correctas.
3. Entrenar: `python src/train_yolo.py`
4. Inferir: `python src/inferencia.py`
