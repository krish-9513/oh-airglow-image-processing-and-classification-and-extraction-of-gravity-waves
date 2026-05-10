import os
import cv2

# ============ CHANGE THESE PATHS ============
INPUT_FOLDER = "/Users/krishmithaiwala/PycharmProjects/DHIRAJ_SIR_PRL/data/differencing_png/FEB/night01"
OUTPUT_VIDEO = "/Users/krishmithaiwala/PycharmProjects/DHIRAJ_SIR_PRL/models/night01_wave_video.mp4"
FPS = 5   # 3–8 recommended (slow motion waves)
# ===========================================


# ---------- LOAD FILES ----------
files = sorted([
    f for f in os.listdir(INPUT_FOLDER)
    if f.lower().endswith(".png")
])

if len(files) == 0:
    raise RuntimeError("No PNG images found!")

print(f"Found {len(files)} frames")

# ---------- READ FIRST IMAGE ----------
first_frame = cv2.imread(os.path.join(INPUT_FOLDER, files[0]), cv2.IMREAD_GRAYSCALE)
height, width = first_frame.shape

# ---------- VIDEO WRITER ----------
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # works on macOS
video = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    FPS,
    (width, height),
    isColor=False
)

# ---------- WRITE FRAMES ----------
for f in files:
    frame = cv2.imread(
        os.path.join(INPUT_FOLDER, f),
        cv2.IMREAD_GRAYSCALE
    )

    if frame is None:
        continue

    video.write(frame)
    print(f"Added frame: {f}")

video.release()

print(f"\n✅ Video saved at: {OUTPUT_VIDEO}")
