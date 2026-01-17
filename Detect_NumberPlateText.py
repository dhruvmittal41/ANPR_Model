import cv2
import re
import os
import pytesseract
from ultralytics import YOLO

# ---------------- CONFIG ----------------
MODEL_PATH = "runs/anpr_yolov8/weights/best.pt"
IMAGE_PATH = "/Users/namanmittal/ANPR_System/dataset/images/train/WB28.jpg"
CONF_THRESH = 0.3

# If tesseract is not in PATH (macOS usually fine)
# pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

# ----------------------------------------

model = YOLO(MODEL_PATH)

# ---------------- TEXT CLEANING ----------------
def clean_text(text):
    text = text.upper()
    text = re.sub(r'[^A-Z0-9]', '', text)

    subs = {
        'O': '0',
        'I': '1',
        'Z': '2',
        'S': '5',
        'B': '8'
    }
    for k, v in subs.items():
        text = text.replace(k, v)

    return text

def find_plate(text):
    """
    Indian number plate patterns:
    KA01AB1234
    DL8CAF5031
    """
    match = re.search(r'[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{3,4}', text)
    return match.group() if match else None

# ---------------- IMAGE PREPROCESS ----------------
def preprocess_plate(plate):
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    _, thresh = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return thresh

# ---------------- OCR ----------------
def ocr_plate(img):
    config = (
        "--oem 3 "
        "--psm 7 "
        "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )
    text = pytesseract.image_to_string(img, config=config)
    return text.strip()

# ---------------- MAIN ----------------
def main():
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print("❌ Image not found")
        return

    results = model.predict(img, conf=CONF_THRESH)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate = img[y1:y2, x1:x2]

            if plate.size == 0:
                continue

            plate_proc = preprocess_plate(plate)

            raw_text = ocr_plate(plate_proc)
            cleaned = clean_text(raw_text)
            plate_text = find_plate(cleaned)

            if plate_text:
                print("✅ VALID PLATE:", plate_text)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    img, plate_text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2
                )
            else:
                print("⚠ OCR RAW:", raw_text, "→", cleaned)

            # Debug windows
            cv2.imshow("Plate", plate)
            cv2.imshow("Processed", plate_proc)
            cv2.waitKey(0)

    cv2.imshow("Final Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
