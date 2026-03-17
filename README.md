# Taller de Detección de Casas e Inpainting con YOLOv8

## 1. Descripción del dataset y origen de imágenes

El conjunto de datos está compuesto originalmente por 58 fotografías de fachadas de casas tomadas en diversas ciudades de Colombia. Las imágenes fueron capturadas a diferentes horas del día y desde distintos ángulos para maximizar la variabilidad visual.

Las imágenes fueron anotadas de forma manual con la herramienta Roboflow, identificando dos tipos de objetos:
- **casa** — fachada visible de la vivienda
- **poste** — postes de luz o similares en la escena

El dataset se almacena en Roboflow bajo el proyecto **`proyecto_casas_y_postes`** (versión 2) y se exporta en formato **YOLOv8**.

---

## 2. Estructura del proyecto

```
taller-yolo-casas/
├── src/
│   ├── download_dataset.py   # Descarga el dataset desde Roboflow (v2)
│   ├── train_yolo.py         # Entrena YOLOv8m (100 epochs, AdamW, augmentaciones avanzadas)
│   ├── export_model.py       # Exporta el modelo a ONNX (opcional)
│   ├── inferencia.py         # Ejecuta inferencia sobre imágenes nuevas
│   ├── inpainting.py         # Detecta postes → crea máscara → LaMa inpainting
│   └── val_metrics.py        # Evalúa el modelo con métricas por clase
├── models/
│   └── best_colab.pt         # Modelo pre-entrenado en Colab (listo para usar)
├── .env                      # Credenciales de Roboflow
├── requirements.txt          # Dependencias
└── README.md                 # Este archivo
```

---

## 3. Requisitos previos

- Python 3.9+
- `pip` o `pip3`

---

## 4. Instalación

```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate   # Linux/Mac

# Instalar dependencias
pip install -r requirements.txt
```

---

## 5. Ejecución rápida (modelo ya incluido)

El modelo pre-entrenado (`models/best_colab.pt`) ya está incluido en el repositorio, por lo que **no es necesario entrenar**. Solo se necesitan 3 pasos:

| # | Paso | Comando |
|---|------|---------|
| 1 | Descargar dataset (imágenes de prueba) | `python src/download_dataset.py` |
| 2 | Inferencia (detectar casas y postes) | `python src/inferencia.py` |
| 3 | Inpainting (eliminar postes) | `python src/inpainting.py` |

Los resultados se guardan en la carpeta `resultados/`.

---

## 6. Guía detallada paso a paso

### Paso 1: Descargar el dataset

```bash
python src/download_dataset.py
```

Descarga el dataset versión **2** del proyecto `proyecto_casas_y_postes` en formato **YOLOv8** a la carpeta `dataset/`. Las imágenes de validación se usan para probar la inferencia y el inpainting.

### Paso 2: Inferencia (detectar casas y postes)

```bash
python src/inferencia.py
```

Usa el modelo `models/best_colab.pt` para detectar casas y postes en las imágenes de validación. Guarda las predicciones visuales (con cajas de detección) en `resultados/`.

### Paso 3: Inpainting (eliminar postes)

```bash
python src/inpainting.py [ruta_imagen_opcional]
```

Proceso:
1. Carga el modelo `models/best_colab.pt`
2. Detecta postes en la imagen
3. Crea máscara binaria (solo postes)
4. Dilata la máscara (kernel 2×2) para cubrir bordes
5. Aplica **LaMa inpainting** para eliminar los postes
6. Guarda el resultado en `resultados/`

Si no se pasa una imagen, usa automáticamente la primera imagen de `dataset/valid/images/`.

---

## 7. Modelo pre-entrenado

El archivo `models/best_colab.pt` contiene el modelo entrenado en **Google Colab** con GPU Tesla T4. Fue entrenado con **YOLOv8 Medium** (`yolov8m.pt`), 100 epochs, optimizador AdamW, y augmentaciones avanzadas. Resultados:

| Clase | Precision | Recall | mAP50 | mAP50-95 |
|-------|-----------|--------|-------|----------|
| casa  | 0.736     | 0.296  | 0.473 | 0.299    |
| poste | 0.783     | 0.318  | 0.578 | 0.325    |
| **all** | **0.759** | **0.307** | **0.525** | **0.312** |

---

## 8. Scripts adicionales (opcionales)

Estos scripts solo son necesarios si se desea **re-entrenar** el modelo o evaluarlo:

| Script | Descripción |
|--------|-------------|
| `python src/train_yolo.py` | Re-entrena el modelo desde cero (requiere GPU) |
| `python src/val_metrics.py` | Evalúa métricas por clase (casa, poste) |
| `python src/export_model.py` | Exporta a ONNX para producción |
