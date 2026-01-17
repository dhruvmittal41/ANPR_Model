import os
import random
import shutil

# ===== CONFIG =====
RAW_IMAGES = "dataset_raw/images/"
RAW_LABELS = "dataset_raw/labels/"
OUT_DIR = "dataset"

TRAIN_RATIO = 0.8
SEED = 42
# ==================

random.seed(SEED)

# Create output directories
for split in ["train", "val"]:
    os.makedirs(os.path.join(OUT_DIR, "images", split), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "labels", split), exist_ok=True)

# Get image files
image_files = [
    f for f in os.listdir(RAW_IMAGES)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

print(f"📸 Found {len(image_files)} images")

# Keep only images that have labels
paired_images = []
for img in image_files:
    label = os.path.splitext(img)[0] + ".txt"
    if os.path.exists(os.path.join(RAW_LABELS, label)):
        paired_images.append(img)

print(f"🧾 Images with labels: {len(paired_images)}")

# Shuffle
random.shuffle(paired_images)

# Split
split_idx = int(len(paired_images) * TRAIN_RATIO)
train_imgs = paired_images[:split_idx]
val_imgs = paired_images[split_idx:]

def copy_pair(img_name, split):
    shutil.copy(
        os.path.join(RAW_IMAGES, img_name),
        os.path.join(OUT_DIR, "images", split, img_name)
    )
    shutil.copy(
        os.path.join(RAW_LABELS, os.path.splitext(img_name)[0] + ".txt"),
        os.path.join(OUT_DIR, "labels", split, os.path.splitext(img_name)[0] + ".txt")
    )

# Copy files
for img in train_imgs:
    copy_pair(img, "train")

for img in val_imgs:
    copy_pair(img, "val")

print("✅ Train/Val split completed")
print(f"📊 Train images: {len(train_imgs)}")
print(f"📊 Val images: {len(val_imgs)}")
