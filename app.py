from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO
import torch
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS untuk React frontend (izinkan semua origin untuk dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Check device (CUDA jika available)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Using device: {device} (CUDA available: {torch.cuda.is_available()})")

# Load model YOLO (sekali di startup)
try:
    model = YOLO("models/bestyolo11n_new.pt")  # Ganti path jika perlu
    model.to(device)
   
    logger.info(f"Model YOLO loaded on {device}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None

# Class names untuk BISINDO
CLASS_NAMES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
]

# Model untuk request body (base64 image)
class ImageRequest(BaseModel):
    image: str

@app.post("/detect")
async def detect(request: ImageRequest):
    if not model:
        return {"error": "Model not loaded."}, 500

    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image.split(',')[1])
        image = Image.open(BytesIO(image_data))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        logger.error(f"Image processing error: {e}")
        return {"error": f"Failed to process image: {str(e)}"}, 400

    # Inferensi YOLO 
    results = model(frame, verbose=False, imgsz=640, conf=0.6)

    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            if conf > 0.6:
                detections.append({
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "score": conf,
                    "classId": cls,
                    "className": CLASS_NAMES[cls]
                })

    return detections

# WebSocket untuk streaming real-time (kirim frame, terima detections)
@app.websocket("/ws/detect")
async def websocket_detect(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            if 'image' not in data:
                await websocket.send_json({"error": "No image in request"})
                continue

            # Process seperti di /detect
            try:
                image_data = base64.b64decode(data['image'].split(',')[1])
                image = Image.open(BytesIO(image_data))
                frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            except Exception as e:
                await websocket.send_json({"error": f"Image processing error: {str(e)}"})
                continue

            results = model(frame, verbose=False, imgsz=640, conf=0.5)  # Hilangkan half
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    if conf > 0.6:
                        detections.append({
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "score": conf,
                            "classId": cls,
                            "className": CLASS_NAMES[cls]
                        })

            await websocket.send_json(detections)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")