import os
import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

def load_csv_image(path):
    try:
        # Quick file size check
        if os.path.getsize(path) == 0:
            raise EmptyDataError("Empty file")

        df = pd.read_csv(path, header=None)

        if df.empty:
            raise EmptyDataError("No data in file")

        img = df[0].str.split(';', expand=True)
        img = img.apply(pd.to_numeric, errors='coerce')
        img = img.values.astype(np.float32)

        # Remove NaN / inf
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

        return img

    except EmptyDataError:
        print(f"[SKIPPED] Empty or invalid CSV: {path}")
        return None

    except Exception as e:
        print(f"[ERROR] Failed to read {path} → {e}")
        return None


def save_npy(csv_path, npy_path):
    img = load_csv_image(csv_path)

    if img is None:
        return  # safely skip

    os.makedirs(os.path.dirname(npy_path), exist_ok=True)
    np.save(npy_path, img)
    print(f"[SAVED] {npy_path}")


# ================= RAW IMAGES =================
RAW_CSV_ROOT = "/Users/krishmithaiwala/PycharmProjects/DHIRAJ_SIR_PRL/data/raw/FEB"
RAW_NPY_ROOT = "data/raw_npy/FEB"

for night in sorted(os.listdir(RAW_CSV_ROOT)):
    night_csv = os.path.join(RAW_CSV_ROOT, night)
    if not os.path.isdir(night_csv):
        continue

    print(f"\nProcessing CSVs in {night}")

    for f in sorted(os.listdir(night_csv)):
        if f.endswith(".csv"):
            save_npy(
                os.path.join(night_csv, f),
                os.path.join(RAW_NPY_ROOT, night, f.replace(".csv", ".npy"))
            )

# ================= DARK & FLAT =================
save_npy(
    "data/calibration/dark_master.csv",
    "data/calibration/dark_master.npy"
)

save_npy(
    "data/calibration/flat_master.csv",
    "data/calibration/flat_master.npy"
)
