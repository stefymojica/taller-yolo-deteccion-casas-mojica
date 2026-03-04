import io
import os
import glob
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
from PIL import Image

app = FastAPI(
    title="API deteccion de casas",
    description="API para detectar casas en imágenes usando YOLO26 Medium",
    version="1.0.0",
)

# Use ONNX model directly for production inference
# Get the directory of this script to build absolute path
script_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(script_dir, "..", "models", "house_detector_prod.onnx")
print(f"--- Iniciando API con modelo: {MODEL_PATH} ---")
model = YOLO(MODEL_PATH)


@app.post("/predict", summary="Detectar casas en imagen")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, detail="El archivo enviado no es una imagen."
        )

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_array = np.array(image)

        results = model.predict(img_array, conf=0.25, imgsz=640)

        annotated_frame = results[0].plot()

        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        res_image = Image.fromarray(annotated_frame_rgb)

        img_byte_arr = io.BytesIO()
        res_image.save(img_byte_arr, format="JPEG")
        img_byte_arr.seek(0)

        return StreamingResponse(img_byte_arr, media_type="image/jpeg")

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error procesando la imagen: {str(e)}"
        )


@app.get("/", include_in_schema=False)
async def root():
    return {"message": "API de Detección de Casas activa. Ve a /docs para probarla."}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
