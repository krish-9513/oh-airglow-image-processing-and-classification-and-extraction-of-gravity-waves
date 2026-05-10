import os
import numpy as np
import re

# ============ CHANGE THESE PATHS ============
INPUT_FOLDER = "/Users/krishmithaiwala/PycharmProjects/DHIRAJ_SIR_PRL/data/classified/FEB/night01"
OUTPUT_FOLDER = "data/differencing/FEB/night01"
DELTA_T = 60.0  # seconds
# ============================================


def time_from_filename(fname):
    """
    Extract time in seconds from filename like:
    KOP00-01-12.32.npy
    """
    # Strict pattern: stop BEFORE ".npy"
    match = re.search(r"KOP(\d+)-(\d+)-(\d+\.\d+)\.npy$", fname)
    if not match:
        return None

    hh = int(match.group(1))
    mm = int(match.group(2))
    ss = float(match.group(3))

    return hh * 3600 + mm * 60 + ss


# ---------- SETUP ----------
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

files = []
for f in os.listdir(INPUT_FOLDER):
    if f.endswith(".npy"):
        t = time_from_filename(f)
        if t is not None:
            files.append((f, t))

# Sort by time
files.sort(key=lambda x: x[1])

# Build lookup table
time_to_file = {t: f for f, t in files}
times = sorted(time_to_file.keys())

print(f"Found {len(times)} valid frames")

# ---------- DIFFERENCING ----------
for t in times:
    t2 = t + DELTA_T

    if t2 not in time_to_file:
        continue  # skip if matching frame not found

    f1 = time_to_file[t]
    f2 = time_to_file[t2]

    img1 = np.load(os.path.join(INPUT_FOLDER, f1))
    img2 = np.load(os.path.join(INPUT_FOLDER, f2))

    diff = img2 - img1

    out_name = f2.replace(
        ".npy",
        f"_minus_{f1.replace('.npy','')}.npy"
    )

    np.save(
        os.path.join(OUTPUT_FOLDER, out_name),
        diff
    )

    print(f"{f2} − {f1} → saved")
