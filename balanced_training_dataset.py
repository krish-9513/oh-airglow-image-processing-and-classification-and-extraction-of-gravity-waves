import os
import random
import shutil

# ================= CONFIG =================
SOURCE_DIR = "/Users/krishmithaiwala/PycharmProjects/DHIRAJ_SIR_PRL/data/enhanced_train"
OUTPUT_DIR = "/Users/krishmithaiwala/PycharmProjects/DHIRAJ_SIR_PRL/data/balanced_train"

TARGET_PER_CLASS = 7000
SEED = 42
# ==========================================

random.seed(SEED)

os.makedirs(OUTPUT_DIR, exist_ok=True)

classes = os.listdir(SOURCE_DIR)

print("Balancing dataset...\n")

for cls in classes:

    src_cls_path = os.path.join(SOURCE_DIR, cls)

    if not os.path.isdir(src_cls_path):
        continue

    dst_cls_path = os.path.join(OUTPUT_DIR, cls)
    os.makedirs(dst_cls_path, exist_ok=True)

    images = [f for f in os.listdir(src_cls_path) if f.lower().endswith(".png")]

    count = len(images)

    print(f"Class: {cls}")
    print(f"Original count: {count}")

    if count >= TARGET_PER_CLASS:
        # ✅ UNDERSAMPLE
        selected = random.sample(images, TARGET_PER_CLASS)

    else:
        # ✅ OVERSAMPLE
        selected = images.copy()

        while len(selected) < TARGET_PER_CLASS:
            selected.append(random.choice(images))

    print(f"Balanced count: {len(selected)}")

    # Copy files
    for i, img_name in enumerate(selected):
        src_img = os.path.join(src_cls_path, img_name)

        # Rename duplicates safely
        dst_img = os.path.join(dst_cls_path, f"{i:05d}_{img_name}")

        shutil.copy(src_img, dst_img)

    print()

print("✅ Dataset balancing complete")
print("📁 Saved to:", OUTPUT_DIR)
