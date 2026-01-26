import os
import shutil

# -------- CONFIG --------
IMAGE_DIR = "ocr_dataset/images"
LABEL_FILE = "ocr_dataset/labels.txt"
PREFIX = "img_"
# ------------------------

# ---------- BACKUP ----------
shutil.copy(LABEL_FILE, LABEL_FILE + ".backup")
print("📦 labels.txt backed up")

# ---------- READ LABELS ----------
entries = []  # (old_name, label_or_empty)

with open(LABEL_FILE, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            entries.append((parts[0], parts[1]))
        else:
            entries.append((parts[0], ""))

print(f"📄 Loaded {len(entries)} label entries")

# ---------- RENAME IMAGES + UPDATE LABELS ----------
new_entries = []

for old_name, label in entries:
    old_path = os.path.join(IMAGE_DIR, old_name)

    if not os.path.exists(old_path):
        print(f"⚠️ Missing image: {old_name}")
        continue

    new_name = PREFIX + old_name
    new_path = os.path.join(IMAGE_DIR, new_name)

    os.rename(old_path, new_path)

    new_entries.append((new_name, label))

# ---------- WRITE UPDATED LABEL FILE ----------
with open(LABEL_FILE, "w") as f:
    for name, label in new_entries:
        if label:
            f.write(f"{name} {label}\n")
        else:
            f.write(f"{name}\n")

print("✅ Images renamed")
print("📄 labels.txt updated safely")
