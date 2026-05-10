import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import shutil

# ================= CONFIG =================
MODEL_PATH = "/Users/krishmithaiwala/PycharmProjects/DHIRAJ_SIR_PRL/models/cnn_v3/model1.h5"
INPUT_DIR  = "/Users/krishmithaiwala/PycharmProjects/DHIRAJ_SIR_PRL/data/processed"
OUTPUT_DIR = "/Users/krishmithaiwala/PycharmProjects/DHIRAJ_SIR_PRL/data/classified_processed_file"

IMG_HEIGHT = 256
IMG_WIDTH  = 320

CONF_THRESHOLD = 0.6
# ==========================================

model = load_model(MODEL_PATH)
print("✅ Model loaded")

# EXACT class order from training
class_names = ['clear', 'cloudy', 'glare', 'moon']

# ---------- Statistics Matrix ----------
stats = {cls: {"high": 0, "low": 0} for cls in class_names}

def prepare_image(img_path):
    img = image.load_img(
        img_path,
        color_mode="grayscale",
        target_size=(IMG_HEIGHT, IMG_WIDTH)
    )

    img_array = image.img_to_array(img)

    # DO NOT divide by 255 (model already rescales)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# ---------- Classification ----------
for month in os.listdir(INPUT_DIR):
    month_path = os.path.join(INPUT_DIR, month)

    if not os.path.isdir(month_path):
        continue

    print(f"\n📂 Processing month: {month}")

    for night in os.listdir(month_path):
        night_path = os.path.join(month_path, night)

        if not os.path.isdir(night_path):
            continue

        print(f"   🌙 Night: {night}")

        for cls in class_names:
            os.makedirs(os.path.join(OUTPUT_DIR, month, night, cls), exist_ok=True)

        for file in os.listdir(night_path):

            if not file.lower().endswith(".png"):
                continue

            img_path = os.path.join(night_path, file)

            try:
                img_array = prepare_image(img_path)

                preds = model.predict(img_array, verbose=0)[0]

                predicted_index = np.argmax(preds)
                predicted_class = class_names[predicted_index]
                confidence = float(preds[predicted_index])

                dest_path = os.path.join(
                    OUTPUT_DIR, month, night, predicted_class, file
                )

                shutil.copy(img_path, dest_path)

                # ---------- Confidence Statistics ----------
                if confidence >= CONF_THRESHOLD:
                    stats[predicted_class]["high"] += 1
                else:
                    stats[predicted_class]["low"] += 1
                    print(f"⚠️ Low confidence → {file} → {predicted_class} ({confidence:.2f})")

            except Exception as e:
                print("❌ Error processing:", img_path)
                print(e)

# ---------- Save Confidence Matrix to CSV ----------
csv_path = os.path.join(OUTPUT_DIR, "confidence_matrix.csv")

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Class", "High Confidence", "Low Confidence"])

    for cls in class_names:
        writer.writerow([
            cls,
            stats[cls]["high"],
            stats[cls]["low"]
        ])

print("\n📄 Confidence matrix saved to:", csv_path)

# ---------- Heatmap Generation ----------
matrix = []
for cls in class_names:
    matrix.append([stats[cls]["high"], stats[cls]["low"]])

matrix = np.array(matrix)

plt.figure(figsize=(6, 4))
plt.imshow(matrix)

plt.xticks([0, 1], ["High Confidence", "Low Confidence"])
plt.yticks(range(len(class_names)), class_names)

plt.title("Confidence Statistics Matrix")
plt.colorbar()

for i in range(len(class_names)):
    for j in range(2):
        plt.text(j, i, matrix[i, j], ha="center", va="center")

heatmap_path = os.path.join(OUTPUT_DIR, "confidence_heatmap.png")
plt.tight_layout()
plt.savefig(heatmap_path)
plt.close()

print("🖼 Heatmap saved to:", heatmap_path)

print("\n================ CONFIDENCE MATRIX ================")
for cls in class_names:
    print(f"{cls:10s} → High: {stats[cls]['high']:5d} | Low: {stats[cls]['low']:5d}")
print("===================================================")

print("\n✅ Classification completed successfully")
print("📁 Output saved at:", OUTPUT_DIR)
