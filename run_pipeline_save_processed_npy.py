import os
import numpy as np

from preprocessing.cosmic import remove_cosmic_rays
from preprocessing.calibration import dark_correction, flat_field_correction
from preprocessing.regrid import regrid_image

# ================= PATHS =================
RAW_ROOT = "/Users/krishmithaiwala/PycharmProjects/DHIRAJ_SIR_PRL/data/raw_npy/FEB"

DARK = np.load(
    "/Users/krishmithaiwala/PycharmProjects/DHIRAJ_SIR_PRL/data/calibration/darkframe.npy"
).astype(np.float32)

FLAT = np.load(
    "/Users/krishmithaiwala/PycharmProjects/DHIRAJ_SIR_PRL/data/calibration/flatfieldframe.npy"
).astype(np.float32)

OUT_ROOT = "data/processed_npy/FEB"

CNN_SHAPE = (256, 320)
# ========================================


# ================= PIPELINE =================
for night in sorted(os.listdir(RAW_ROOT)):
    night_path = os.path.join(RAW_ROOT, night)
    if not os.path.isdir(night_path):
        continue

    print(f"\nProcessing {night}")

    # Output folder: processed_npy/FEB/nightXX
    out_dir = os.path.join(OUT_ROOT, night)
    os.makedirs(out_dir, exist_ok=True)

    for f in sorted(os.listdir(night_path)):
        if not f.endswith(".npy"):
            continue

        # ---------- LOAD RAW IMAGE ----------
        img = np.load(os.path.join(night_path, f)).astype(np.float32)

        # ---------- 1. COSMIC RAY REMOVAL ----------
        img = remove_cosmic_rays(img)

        # ---------- 2. MATCH DARK & FLAT TO RAW GEOMETRY ----------
        dark_rs = regrid_image(DARK, target_shape=img.shape)
        flat_rs = regrid_image(FLAT, target_shape=img.shape)

        # ---------- 3. APPLY CALIBRATION ----------
        img = dark_correction(img, dark_rs)
        img = flat_field_correction(img, flat_rs)

        # ---------- 4. REGRID TO FINAL SIZE (256 × 320) ----------
        img = regrid_image(img, target_shape=CNN_SHAPE)

        # ---------- SAVE PROCESSED NPY ----------
        out_name = f.replace(".npy", ".npy")  # keep same name
        np.save(
            os.path.join(out_dir, out_name),
            img
        )

        print(f"  {f} → saved to {out_dir}")
