import os
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

# ============ PATHS ============
INPUT_FOLDER = "/Users/krishmithaiwala/PycharmProjects/DHIRAJ_SIR_PRL/data/differencing/FEB/night01"
OUTPUT_FOLDER = "data/differencing_png/FEB/night01"
# ===============================


def highpass_filter(img, sigma=15):
    """
    Remove large-scale background, keep wave structures
    """
    low_freq = gaussian_filter(img, sigma=sigma)
    return img - low_freq


def wave_contrast(img, p=98):
    """
    Symmetric contrast stretch for DIFFERENCE images
    """
    vmax = np.percentile(np.abs(img), p)
    img = np.clip(img, -vmax, vmax)
    img = (img + vmax) / (2 * vmax + 1e-8)  # map to [0,1]
    return img


# ---------- SETUP ----------
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

files = sorted([f for f in os.listdir(INPUT_FOLDER) if f.endswith(".npy")])

print(f"Found {len(files)} difference images")

# ---------- CONVERT TO PNG ----------
for f in files:
    diff = np.load(os.path.join(INPUT_FOLDER, f)).astype(np.float32)

    # 1. High-pass filter (CRITICAL for waves)
    diff_hp = highpass_filter(diff, sigma=15)

    # 2. Symmetric contrast stretch
    diff_vis = wave_contrast(diff_hp, p=98)

    out_name = f.replace(".npy", ".png")

    cv2.imwrite(
        os.path.join(OUTPUT_FOLDER, out_name),
        (diff_vis * 255).astype("uint8")
    )

    print(f"Saved {out_name}")
