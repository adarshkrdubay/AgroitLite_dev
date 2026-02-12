import os
import zipfile
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import subprocess

# ===============================
# CONFIG
# ===============================
DATASET_NAME = "emmarex/plantdisease"
DATASET_ZIP = "plantdisease.zip"
EXTRACT_PATH = "plantvillage"

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15

CROPS = {
    "Mango": ["Mango___Anthracnose",
              "Mango___Healthy",
              "Mango___Powdery_Mildew"],

    "Tomato": ["Tomato___Early_blight",
               "Tomato___Late_blight",
               "Tomato___Healthy"],

    "Potato": ["Potato___Early_blight",
               "Potato___Late_blight",
               "Potato___Healthy"]
}

# ===============================
# STEP 1 — DOWNLOAD DATASET
# ===============================
def download_dataset():

    if os.path.exists(EXTRACT_PATH):
        print("Dataset already downloaded.")
        return

    print("Downloading PlantVillage dataset from Kaggle...")
    subprocess.run(["kaggle", "datasets", "download", "-d", DATASET_NAME])

    print("Extracting dataset...")
    with zipfile.ZipFile(DATASET_ZIP, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_PATH)

    print("Dataset extracted.")

# ===============================
# STEP 2 — AGROITLITE MODEL
# ===============================
def build_agroitlite(num_classes):

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    x = layers.Conv2D(16, (3,3), padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.DepthwiseConv2D((3,3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(32, (1,1), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.DepthwiseConv2D((3,3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(64, (1,1), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.GlobalAveragePooling2D()(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(inputs, outputs)

# ===============================
# STEP 3 — TRAIN PER CROP
# ===============================
def train_crop(crop_name, class_list):

    print(f"\n======================")
    print(f"Training for {crop_name}")
    print("======================")

    base_path = os.path.join(EXTRACT_PATH)

    temp_crop_path = f"temp_{crop_name}"
    os.makedirs(temp_crop_path, exist_ok=True)

    # Copy only required classes
    for cls in class_list:
        src = os.path.join(base_path, cls)
        dst = os.path.join(temp_crop_path, cls)
        if not os.path.exists(dst):
            os.system(f"xcopy \"{src}\" \"{dst}\" /E /I /Y")

    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    train_gen = datagen.flow_from_directory(
        temp_crop_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training"
    )

    val_gen = datagen.flow_from_directory(
        temp_crop_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation"
    )

    num_classes = len(train_gen.class_indices)

    model = build_agroitlite(num_classes)

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=[EarlyStopping(patience=3, restore_best_weights=True)],
        verbose=1
    )

    preds = model.predict(val_gen)
    y_true = val_gen.classes
    y_pred = np.argmax(preds, axis=1)

    acc = accuracy_score(y_true, y_pred)

    print(f"{crop_name} Validation Accuracy: {acc*100:.2f}%")

    model.save(f"AgroitLite_{crop_name}.h5")

    return {
        "Crop": crop_name,
        "Classes": num_classes,
        "Accuracy (%)": round(acc*100, 2),
        "Parameters": model.count_params()
    }

# ===============================
# MAIN
# ===============================
if __name__ == "__main__":

    download_dataset()

    results = []

    for crop_name, class_list in CROPS.items():
        results.append(train_crop(crop_name, class_list))

    print("\n===== FINAL SCALABILITY RESULTS =====")
    for r in results:
        print(r)
