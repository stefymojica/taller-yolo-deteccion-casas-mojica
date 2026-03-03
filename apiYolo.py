from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()

# 🔹 Cargar modelo (cambia la ruta a tu .pt)
model = YOLO("modelo_entrenado.pt")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    
    # Leer imagen
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Inferencia
    results = model(image)

    # Dibujar bounding boxes
    annotated_image = results[0].plot()

    # Convertir numpy array a imagen
    output_image = Image.fromarray(annotated_image)

    img_io = io.BytesIO()
    output_image.save(img_io, format="JPEG")
    img_io.seek(0)

    return StreamingResponse(img_io, media_type="image/jpeg")