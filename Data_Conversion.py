import os
import xml.etree.ElementTree as ET

# CHANGE THESE PATHS
XML_DIR = "images/train/video_images"
IMAGE_DIR = "images/train/video_images"
LABEL_DIR = "labels/train"

os.makedirs(LABEL_DIR, exist_ok=True)

CLASS_ID = 0  # number_plate

for xml_file in os.listdir(XML_DIR):
    if not xml_file.endswith(".xml"):
        continue

    tree = ET.parse(os.path.join(XML_DIR, xml_file))
    root = tree.getroot()

    filename = root.find("filename").text
    img_width = int(root.find("size/width").text)
    img_height = int(root.find("size/height").text)

    label_file = os.path.splitext(filename)[0] + ".txt"
    label_path = os.path.join(LABEL_DIR, label_file)

    with open(label_path, "w") as f:
        for obj in root.findall("object"):
            bbox = obj.find("bndbox")

            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)

            x_center = ((xmin + xmax) / 2) / img_width
            y_center = ((ymin + ymax) / 2) / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height

            f.write(f"{CLASS_ID} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

print("✅ XML to YOLO conversion completed")
