import os
import cv2
import numpy as np
from tqdm import tqdm

# ================= CONFIG =================
INPUT_ROOT  = "/Users/krishmithaiwala/PycharmProjects/DHIRAJ_SIR_PRL/data/training"
OUTPUT_ROOT = "/Users/krishmithaiwala/PycharmProjects/DHIRAJ_SIR_PRL/data/enhanced_train"

CLASSES = ["cloudy", "glare", "moon", "clear"]

LOW_PERC  = 1.0
HIGH_PERC = 99.0

CLAHE_CLIP = 3.0
CLAHE_TILE = (8, 8)
# =========================================


def percentile_contrast(img, low=1, high=99):
    """Percentile-based contrast stretching"""
    p_low, p_high = np.percentile(img, (low, high))

    if p_high - p_low < 1e-6:
        return np.zeros_like(img, dtype=np.float32)

    img = np.clip(img, p_low, p_high)
    img = (img - p_low) / (p_high - p_low)
    return img


def apply_clahe(img_norm):
    """Apply CLAHE on normalized image"""
    img_8u = (img_norm * 255).astype(np.uint8)

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP,
        tileGridSize=CLAHE_TILE
    )

    return clahe.apply(img_8u)


def enhance_image(img_path):
    """Load → Enhance → Return enhanced image"""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return None

    img = img.astype(np.float32)

    img_pc = percentile_contrast(img, LOW_PERC, HIGH_PERC)
    img_enhanced = apply_clahe(img_pc)

    return img_enhanced


# ================= MAIN ===================
if __name__ == "__main__":

    print("🔹 Starting dataset enhancement...")

    for cls in CLASSES:
        input_dir  = os.path.join(INPUT_ROOT, cls)
        output_dir = os.path.join(OUTPUT_ROOT, cls)

        os.makedirs(output_dir, exist_ok=True)

        images = [
            f for f in os.listdir(input_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        print(f"\nProcessing class: {cls} ({len(images)} images)")

        for img_name in tqdm(images):
            in_path  = os.path.join(input_dir, img_name)
            out_path = os.path.join(output_dir, img_name)

            enhanced = enhance_image(in_path)

            if enhanced is None:
                print(f"⚠️ Skipped: {in_path}")
                continue

            cv2.imwrite(out_path, enhanced)

    print("\n✅ Dataset enhancement completed successfully!")
