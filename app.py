import streamlit as st
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from model import CRNN
from utils import ctc_greedy_decode

# ---------------- CONFIG ----------------
YOLO_MODEL_PATH = "runs/anpr_yolov8/weights/best.pt"
OCR_MODEL_PATH = "crnn_plate_ocr.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    yolo = YOLO(YOLO_MODEL_PATH)

    ocr = CRNN().to(device)
    ocr.load_state_dict(torch.load(OCR_MODEL_PATH, map_location=device))
    ocr.eval()

    return yolo, ocr

yolo_model, ocr_model = load_models()

# ---------------- OCR PREPROCESS ----------------
def preprocess_plate(plate_img):
    plate_img = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    plate_img = cv2.resize(plate_img, (128, 32))
    plate_img = plate_img.astype("float32") / 255.0
    plate_img = torch.from_numpy(plate_img).unsqueeze(0).unsqueeze(0)
    return plate_img.to(device)

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="ANPR System", layout="centered")
st.title("🚘 Automatic Number Plate Recognition")

uploaded_file = st.file_uploader(
    "Upload a vehicle image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(img, caption="Uploaded Image", channels="BGR")

    # ---------------- YOLO DETECTION ----------------
    results = yolo_model.predict(img, conf=0.4, verbose=False)

    if len(results[0].boxes) == 0:
        st.error("❌ No number plate detected")
    else:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate_crop = img[y1:y2, x1:x2]

            if plate_crop.size == 0:
                continue

            st.image(plate_crop, caption="Detected Number Plate", channels="BGR")

            # ---------------- OCR ----------------
            plate_tensor = preprocess_plate(plate_crop)

            with torch.no_grad():
                preds = ocr_model(plate_tensor)       # (B, T, C)
                preds = preds.permute(1, 0, 2)        # (T, B, C)
                preds = torch.log_softmax(preds, 2)

                text = ctc_greedy_decode(preds)[0]

            st.success(f"📌 Detected Plate Number: **{text}**")
