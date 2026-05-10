import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report

# ================= CONFIG =================
DATA_DIR = "/Users/krishmithaiwala/PycharmProjects/DHIRAJ_SIR_PRL/data/balanced_train"
MODEL_DIR = "/Users/krishmithaiwala/PycharmProjects/DHIRAJ_SIR_PRL/models/cnn_v3"

IMG_HEIGHT = 256
IMG_WIDTH  = 320

# ⭐ Larger batch reduces CPU overhead (VERY important)
BATCH_SIZE = 12      # Try 12 → if memory issue → use 8

EPOCHS = 15          # Enough for convergence with LR scheduler
SEED = 42
# ==========================================

os.makedirs(MODEL_DIR, exist_ok=True)

print("TensorFlow version:", tf.__version__)
print("GPUs available:", tf.config.list_physical_devices("GPU"))

# ============ DATASETS =====================
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    color_mode="grayscale",
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    color_mode="grayscale",
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
num_classes = len(class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(500).prefetch(AUTOTUNE)
val_ds   = val_ds.prefetch(AUTOTUNE)

# ============ DATA AUGMENTATION ============
data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.03),
    layers.RandomTranslation(0.08, 0.08),
    layers.RandomFlip("horizontal"),
])

# ============ LIGHTWEIGHT CNN ==============
model = models.Sequential([
    layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1)),

    data_augmentation,
    layers.Rescaling(1./255),

    layers.Conv2D(24, 3, activation="relu"),   # ⭐ Reduced filters
    layers.MaxPooling2D(),

    layers.Conv2D(48, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(96, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.GlobalAveragePooling2D(),

    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),

    layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ============ CALLBACKS ====================
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    verbose=1
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True
    ),
    lr_scheduler
]

# ============ TRAIN ========================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ============ SAVE MODEL ===================
model.save(os.path.join(MODEL_DIR, "model1.h5"))

# ============ CLASSIFICATION REPORT ========
y_true = []
y_pred = []

for images, labels in val_ds:
    preds = model.predict(images, verbose=0)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(labels.numpy())

report = classification_report(y_true, y_pred, target_names=class_names)

with open(os.path.join(MODEL_DIR, "classification_report1.txt"), "w") as f:
    f.write(report)

print("\n✅ TRAINING COMPLETE")
print("📁 Saved to:", MODEL_DIR)
