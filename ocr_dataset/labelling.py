# import os
# import shutil

# # -------- CONFIG --------
# IMAGE_DIR = "ocr_dataset/images"
# LABEL_FILE = "ocr_dataset/labels.txt"
# PREFIX = "img_"
# # ------------------------

# # ---------- BACKUP ----------
# shutil.copy(LABEL_FILE, LABEL_FILE + ".backup")
# print("📦 labels.txt backed up")

# # ---------- READ LABELS ----------
# entries = []  # (old_name, label_or_empty)

# with open(LABEL_FILE, "r") as f:
#     for line in f:
#         line = line.strip()
#         if not line:
#             continue

#         parts = line.split(maxsplit=1)
#         if len(parts) == 2:
#             entries.append((parts[0], parts[1]))
#         else:
#             entries.append((parts[0], ""))

# print(f"📄 Loaded {len(entries)} label entries")

# # ---------- RENAME IMAGES + UPDATE LABELS ----------
# new_entries = []

# for old_name, label in entries:
#     old_path = os.path.join(IMAGE_DIR, old_name)

#     if not os.path.exists(old_path):
#         print(f"⚠️ Missing image: {old_name}")
#         continue

#     new_name = PREFIX + old_name
#     new_path = os.path.join(IMAGE_DIR, new_name)

#     os.rename(old_path, new_path)

#     new_entries.append((new_name, label))

# # ---------- WRITE UPDATED LABEL FILE ----------
# with open(LABEL_FILE, "w") as f:
#     for name, label in new_entries:
#         if label:
#             f.write(f"{name} {label}\n")
#         else:
#             f.write(f"{name}\n")

# print("✅ Images renamed")
# print("📄 labels.txt updated safely")

import os

IMAGE_DIR = "ocr_dataset/images"
LABEL_FILE = "ocr_dataset/labels.txt"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

# 1. Read existing labels
existing_entries = {}
existing_images = set()

if os.path.exists(LABEL_FILE):
    with open(LABEL_FILE, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue

            parts = line.split(maxsplit=1)
            img = parts[0]
            existing_images.add(img)
            existing_entries[img] = line

# 2. List all images
all_images = sorted([
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(IMAGE_EXTENSIONS)
])

# 3. Find missing images
new_images = [img for img in all_images if img not in existing_images]

# 4. Append missing images (empty label)
with open(LABEL_FILE, "a") as f:
    for img in new_images:
        f.write(f"{img}\n")

print(f"✅ Existing labels preserved: {len(existing_images)}")
print(f"➕ New images added with empty labels: {len(new_images)}")
