import cv2
import os
from ultralytics import YOLO

# ---------------- CONFIG ----------------
MODEL_PATH = "runs/anpr_yolov8/weights/best.pt"
INPUT_DIR = "/Users/namanmittal/ANPR_System/dataset_raw/images"
OUTPUT_DIR = "ocr_dataset/images"
CONF_THRESH = 0.3

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------------
model = YOLO(MODEL_PATH)

def crop_plates_from_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return 0

    results = model.predict(img, conf=CONF_THRESH, verbose=False)

    base_name = os.path.splitext(os.path.basename(img_path))[0]
    count = 0

    for r in results:
        for i, box in enumerate(r.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            plate = img[y1:y2, x1:x2]
            if plate.size == 0:
                continue

            out_name = f"{base_name}_{count}.jpg"
            out_path = os.path.join(OUTPUT_DIR, out_name)

            cv2.imwrite(out_path, plate)
            count += 1

    return count

def main():
    total = 0

    for file in os.listdir(INPUT_DIR):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(INPUT_DIR, file)
            num = crop_plates_from_image(img_path)
            total += num
            print(f"📸 {file}: {num} plates")

    print(f"\n✅ Total cropped plates saved: {total}")

if __name__ == "__main__":
    main()
