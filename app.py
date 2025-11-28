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

# ===================== SETUP LOGGING =====================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===================== FASTAPI APP =====================
app = FastAPI()

# CORS untuk React frontend (dev: izinkan semua origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================== DEVICE & MODEL LOAD =====================
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device} (CUDA available: {torch.cuda.is_available()})")

try:
    # Load model sekali di startup
    model = YOLO("models/bestyolo11n_new.pt")  # ganti path jika perlu
    model.to(device)

    # Optimasi khusus GPU
    if device == "cuda":
        try:
            # Fuse Conv + BN untuk sedikit speed-up
            model.fuse()
            # Ubah weight ke half precision
            model.model.half()
            # CuDNN autotune untuk GPU
            torch.backends.cudnn.benchmark = True
            logger.info("Enabled fuse + half precision for YOLO model on CUDA")
        except Exception as e:
            # Kalau fuse/half gagal, tetap jalan pakai float32
            logger.warning(f"Failed to enable fuse/half: {e}")

    # Info dtype untuk memastikan
    try:
        sample_param = next(model.model.parameters())
        logger.info(f"Model dtype after init: {sample_param.dtype}")
    except Exception:
        pass

    logger.info(f"Model YOLO loaded on {device}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None

# ===================== LABEL KELAS =====================
CLASS_NAMES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
]

# ===================== REQUEST BODY MODEL =====================
class ImageRequest(BaseModel):
    image: str  # base64 DataURL: "data:image/jpeg;base64,...."

# ===================== REST ENDPOINT /detect =====================
@app.post("/detect")
async def detect(request: ImageRequest):
    if model is None:
        # FastAPI lebih enak pakai Response/HTTPException,
        # tapi biarkan gaya ini kalau frontend-mu sudah cocok.
        return {"error": "Model not loaded."}, 500

    # Decode base64 image
    try:
        image_data = base64.b64decode(request.image.split(",")[1])
        image = Image.open(BytesIO(image_data))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        logger.error(f"Image processing error (REST): {e}")
        return {"error": f"Failed to process image: {str(e)}"}, 400

    # Inferensi YOLO
    try:
        # Ultralytics akan handle dtype (float16/float32) sesuai model
        results = model(frame, verbose=False, imgsz=640, conf=0.6)
    except Exception as e:
        logger.error(f"YOLO inference error (REST): {e}")
        return {"error": f"Inference error: {str(e)}"}, 500

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

# ===================== WEBSOCKET /ws/detect =====================
@app.websocket("/ws/detect")
async def websocket_detect(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket /ws/detect connected")

    try:
        while True:
            # Terima JSON: { "image": "data:image/jpeg;base64,..." }
            data = await websocket.receive_json()
            if "image" not in data:
                await websocket.send_json({"error": "No image in request"})
                continue

            # Decode base64 image (sama seperti REST)
            try:
                image_data = base64.b64decode(data["image"].split(",")[1])
                image = Image.open(BytesIO(image_data))
                frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            except Exception as e:
                err_msg = f"Image processing error (WS): {str(e)}"
                logger.error(err_msg)
                await websocket.send_json({"error": err_msg})
                continue

            if model is None:
                await websocket.send_json({"error": "Model not loaded"})
                continue

            # Inferensi YOLO (dtype already half if on CUDA)
            try:
                results = model(frame, verbose=False, imgsz=640, conf=0.7)
            except Exception as e:
                err_msg = f"Inference error (WS): {str(e)}"
                logger.error(err_msg)
                await websocket.send_json({"error": err_msg})
                continue

            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())

                    if conf > 0.7:
                        detections.append({
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "score": conf,
                            "classId": cls,
                            "className": CLASS_NAMES[cls]
                        })

            # Kirim balik array detections
            await websocket.send_json(detections)

    except WebSocketDisconnect:
        logger.info("WebSocket /ws/detect disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
