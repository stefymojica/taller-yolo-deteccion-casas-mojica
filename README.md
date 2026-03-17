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
│   ├── val_metrics.py        # Evalúa el modelo con métricas por clase
│   ├── inferencia.py         # Ejecuta inferencia sobre imágenes nuevas
│   ├── inpainting.py         # Detecta postes → crea máscara → LaMa inpainting
│   └── export_model.py       # Exporta el modelo a ONNX (opcional)
├── models/
│   └── best_colab.pt         # Modelo pre-entrenado en Colab (backup)
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

El repositorio incluye `models/best_colab.pt`, un modelo **ya entrenado en Google Colab**. Si se desea ver los resultados sin entrenar, solo se necesitan **3 comandos**:

```bash
python src/download_dataset.py       # 1. Descargar imágenes de prueba
python src/inferencia.py             # 2. Detectar casas y postes
python src/inpainting.py             # 3. Eliminar postes con inpainting
```

Los resultados se guardan en la carpeta `resultados/`.

---

## 6. Pipeline completo desde cero (paso a paso)

| # | Paso | Comando |
|---|------|---------|
| 1 | Descargar dataset | `python src/download_dataset.py` |
| 2 | Entrenar modelo | `python src/train_yolo.py` |
| 3 | Evaluar métricas | `python src/val_metrics.py` |
| 4 | Inferencia (detectar casas y postes) | `python src/inferencia.py` |
| 5 | Inpainting (eliminar postes) | `python src/inpainting.py` |
| 6 | Exportar ONNX (opcional) | `python src/export_model.py` |

---

## 7. Guía detallada

### Paso 1: Descargar el dataset

```bash
python src/download_dataset.py
```

Descarga el dataset versión **2** del proyecto `proyecto_casas_y_postes` en formato **YOLOv8** a la carpeta `dataset/`.

### Paso 2: Entrenar el modelo

```bash
python src/train_yolo.py
```

Entrena **YOLOv8 Medium** (`yolov8m.pt`) con la siguiente configuración:

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| `epochs` | 100 | Más épocas con early stopping |
| `patience` | 25 | Espera 25 épocas sin mejora |
| `optimizer` | AdamW | Mejor para datasets pequeños |
| `lr0` | 0.001 | Learning rate inicial |
| `box` | 7.5 | Mayor peso a localización |
| `cls` | 0.3 | Menor peso a clasificación (solo 2 clases) |
| `scale` | 0.4 | Zoom — ayuda con postes lejos y cerca |
| `mosaic` | 0.5 | Combina imágenes (pocas muestras) |

El mejor modelo se guarda en `runs/detect/train_postes_v2/weights/best.pt`.

> **Nota:** Si no se tiene GPU disponible, el repositorio incluye `models/best_colab.pt` — un modelo ya entrenado en Google Colab con GPU Tesla T4. Los scripts de inferencia e inpainting lo usan automáticamente si no encuentran un modelo entrenado localmente.

### Paso 3: Evaluar métricas

```bash
python src/val_metrics.py
```

Evalúa el modelo sobre el set de validación y muestra métricas **por clase** (casa, poste) y **globales**:
- Precision, Recall, mAP@0.5, mAP@0.5:0.95

### Paso 4: Inferencia

```bash
python src/inferencia.py
```

Detecta casas y postes en las imágenes de validación y guarda las predicciones visuales en `resultados/`.

### Paso 5: Inpainting (eliminar postes)

```bash
python src/inpainting.py [ruta_imagen_opcional]
```

Proceso:
1. Carga el mejor modelo YOLO disponible
2. Detecta postes en la imagen
3. Crea máscara binaria (solo postes)
4. Dilata la máscara (kernel 2×2) para cubrir bordes
5. Aplica **LaMa inpainting** para eliminar los postes
6. Guarda el resultado en `resultados/`

Si no se pasa una imagen, usa automáticamente la primera imagen de `dataset/valid/images/`.

---

## 8. Resultados del modelo pre-entrenado

El archivo `models/best_colab.pt` fue entrenado en **Google Colab** con GPU Tesla T4 y logró:

| Clase | Precision | Recall | mAP50 | mAP50-95 |
|-------|-----------|--------|-------|----------|
| casa  | 0.736     | 0.296  | 0.473 | 0.299    |
| poste | 0.783     | 0.318  | 0.578 | 0.325    |
| **all** | **0.759** | **0.307** | **0.525** | **0.312** |
