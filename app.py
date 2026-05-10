import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
import plotly.express as px

from preprocessing.csv_loader import load_csv_image
from preprocessing.cosmic import remove_cosmic_rays
from preprocessing.calibration import dark_correction, flat_field_correction
from preprocessing.regrid import regrid_image

# ---------------- PATH CONFIG ----------------
RAW_DIR = "/Users/krishmithaiwala/PycharmProjects/DHIRAJ_SIR_PRL/data/raw"
CLASSIFIED_DIR = "/Users/krishmithaiwala/PycharmProjects/DHIRAJ_SIR_PRL/data/classified_processed_file"
DB_PATH = "/Users/krishmithaiwala/PycharmProjects/DHIRAJ_SIR_PRL/data/classification_db.csv"
MODEL_PATH = "/Users/krishmithaiwala/PycharmProjects/DHIRAJ_SIR_PRL/models/cnn_v3/model1.h5"

# ---------------- CALIBRATION ----------------
DARK = np.load("data/calibration/darkframe.npy").astype(np.float32)
FLAT = np.load("data/calibration/flatfieldframe.npy").astype(np.float32)

CNN_SHAPE = (256, 320)

MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
          "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

label_map = {0: "clear", 1: "cloudy", 2: "moon", 3: "glare"}

st.set_page_config(page_title="Sky Dashboard", layout="wide")
st.title("🌙 Sky Image Classification Dashboard")

# ---------------- LOAD DATABASE ----------------
if os.path.exists(DB_PATH):
    db = pd.read_csv(DB_PATH)
else:
    db = pd.DataFrame(columns=["year", "month", "day", "filename", "label"])

# ---------------- HELPERS ----------------
def get_years():
    return sorted([
        y for y in os.listdir(RAW_DIR)
        if os.path.isdir(os.path.join(RAW_DIR, y))
    ])

def get_nights(year, month):
    path = os.path.join(RAW_DIR, year, month)
    return sorted([
        n for n in os.listdir(path)
        if os.path.isdir(os.path.join(path, n))
    ])

def extract_day_from_night(night_folder):
    return night_folder.replace("night", "")

def preprocess_csv(csv_path):

    img = load_csv_image(csv_path)
    if img is None:
        return None

    img = remove_cosmic_rays(img)

    dark_rs = regrid_image(DARK, target_shape=img.shape)
    flat_rs = regrid_image(FLAT, target_shape=img.shape)

    img = dark_correction(img, dark_rs)
    img = flat_field_correction(img, flat_rs)

    img = regrid_image(img, target_shape=CNN_SHAPE)

    img = np.expand_dims(img, axis=(0, -1))

    return img

def is_month_classified(year, month):

    if not db[(db["year"] == year) & (db["month"] == month)].empty:
        return True

    if os.path.exists(os.path.join(CLASSIFIED_DIR, year, month)):
        return True

    return False

def count_from_classified_folder(year, month):

    month_path = os.path.join(CLASSIFIED_DIR, year, month)

    if not os.path.exists(month_path):
        return None

    records = []

    for night in sorted(os.listdir(month_path)):

        night_path = os.path.join(month_path, night)

        if not os.path.isdir(night_path):
            continue

        day = night.replace("night", "")

        clear_path = os.path.join(night_path, "clear")

        count = len(os.listdir(clear_path)) if os.path.exists(clear_path) else 0

        records.append({"day": day, "count": count})

    df = pd.DataFrame(records)

    if df.empty:
        return None

    df["day"] = df["day"].astype(int)
    df = df.sort_values("day")

    return df

# ---------------- UI ----------------
years = get_years()

if not years:
    st.error("No data available")
    st.stop()

year = st.selectbox("Select Year", years)
month = st.selectbox("Select Month", MONTHS)

# ---------------- MAIN ACTION ----------------
if st.button("Submit"):

    if is_month_classified(year, month):

        st.success(f"✅ {month} {year} already classified. Showing visualization...")

    else:

        raw_month_path = os.path.join(RAW_DIR, year, month)

        if not os.path.exists(raw_month_path):

            st.error(f"❌ No images available for {month} {year}")
            st.stop()

        with st.spinner(f"Processing {month} {year} data..."):

            model = tf.keras.models.load_model(MODEL_PATH)
            nights = get_nights(year, month)

            new_entries = []

            total_files = sum(
                len([f for f in os.listdir(os.path.join(raw_month_path, n)) if f.endswith(".csv")])
                for n in nights
            )

            progress_bar = st.progress(0)
            processed = 0

            for night in nights:

                day = extract_day_from_night(night)
                night_path = os.path.join(raw_month_path, night)

                for fname in sorted(os.listdir(night_path)):

                    if not fname.endswith(".csv"):
                        continue

                    model_input = preprocess_csv(os.path.join(night_path, fname))

                    if model_input is None:
                        processed += 1
                        progress_bar.progress(processed / total_files)
                        continue

                    pred = model.predict(model_input, verbose=0)
                    idx = np.argmax(pred)

                    new_entries.append({
                        "year": year,
                        "month": month,
                        "day": day,
                        "filename": fname.replace(".csv", ".npy"),
                        "label": label_map[idx]
                    })

                    processed += 1
                    progress_bar.progress(processed / total_files)

            if new_entries:
                updated_df = pd.DataFrame(new_entries)
                updated_db = pd.concat([db, updated_df], ignore_index=True)
                updated_db.to_csv(DB_PATH, index=False)

            st.success("✅ Month preprocessing & classification completed!")

# ---------------- VISUALIZATION ----------------
month_data = db[(db["year"] == year) & (db["month"] == month)]

if not month_data.empty:

    st.subheader(f"📊 Clear Images per Day — {month} {year} (CSV Source)")

    clear_counts = (
        month_data[month_data["label"] == "clear"]
        .groupby("day")
        .size()
        .reset_index(name="count")
        .sort_values("day")
    )

    fig = px.bar(clear_counts, x="day", y="count")
    st.plotly_chart(fig, use_container_width=True)

else:

    folder_counts = count_from_classified_folder(year, month)

    if folder_counts is not None:

        st.subheader(f"📊 Clear Images per Day — {month} {year} (Folder Source)")

        fig = px.bar(folder_counts, x="day", y="count")
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("No classified data found")
