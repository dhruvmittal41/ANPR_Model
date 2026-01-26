import os
import cv2
import xml.etree.ElementTree as ET

# ---------------- CONFIG ----------------
IMAGE_DIR = "dataset_raw/images"
XML_DIR = "xml_data"
OUT_IMG_DIR = "ocr_dataset/images"
LABEL_FILE = "ocr_dataset/labels.txt"

os.makedirs(OUT_IMG_DIR, exist_ok=True)

# ---------------------------------------

def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.find("filename").text
    obj = root.find("object")

    plate_text = obj.find("name").text.strip()

    bbox = obj.find("bndbox")
    x1 = int(bbox.find("xmin").text)
    y1 = int(bbox.find("ymin").text)
    x2 = int(bbox.find("xmax").text)
    y2 = int(bbox.find("ymax").text)

    return filename, plate_text, (x1, y1, x2, y2)

def main():
    idx = 0
    label_lines = []

    for xml_file in os.listdir(XML_DIR):
        if not xml_file.endswith(".xml"):
            continue

        xml_path = os.path.join(XML_DIR, xml_file)
        filename, plate_text, (x1, y1, x2, y2) = parse_xml(xml_path)

        img_path = os.path.join(IMAGE_DIR, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"❌ Missing image: {filename}")
            continue

        plate_crop = img[y1:y2, x1:x2]
        if plate_crop.size == 0:
            continue

        out_name = f"{idx:06d}.jpg"
        out_path = os.path.join(OUT_IMG_DIR, out_name)

        cv2.imwrite(out_path, plate_crop)

        label_lines.append(f"{out_name} {plate_text}")
        idx += 1

        print(f"✅ Saved {out_name} → {plate_text}")

    with open(LABEL_FILE, "w") as f:
        f.write("\n".join(label_lines))

    print(f"\n🎯 Total samples: {idx}")
    print("📄 Labels saved to labels.txt")

if __name__ == "__main__":
    main()
