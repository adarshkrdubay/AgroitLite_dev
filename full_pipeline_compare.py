import os
import shutil
import random
import time
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# ===============================
# CONFIG
# ===============================
ORIGINAL_DATASET = "dataset"
SPLIT_DATASET = "dataset_split"
TRAIN_DIR = os.path.join(SPLIT_DATASET, "train")
TEST_DIR = os.path.join(SPLIT_DATASET, "test")

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5  # Increase to 25+ for paper

# ===============================
# STEP 1 — TRAIN/TEST SPLIT
# ===============================

def split_dataset(train_ratio=0.8):
    if os.path.exists(SPLIT_DATASET):
        print("Split dataset already exists.")
        return

    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    for class_name in os.listdir(ORIGINAL_DATASET):
        class_path = os.path.join(ORIGINAL_DATASET, class_name)
        if not os.path.isdir(class_path):
            continue

        images = os.listdir(class_path)
        random.shuffle(images)

        split_index = int(len(images) * train_ratio)
        train_images = images[:split_index]
        test_images = images[split_index:]

        os.makedirs(os.path.join(TRAIN_DIR, class_name), exist_ok=True)
        os.makedirs(os.path.join(TEST_DIR, class_name), exist_ok=True)

        for img in train_images:
            shutil.copy(os.path.join(class_path, img),
                        os.path.join(TRAIN_DIR, class_name, img))

        for img in test_images:
            shutil.copy(os.path.join(class_path, img),
                        os.path.join(TEST_DIR, class_name, img))

    print("Dataset split complete.")


# ===============================
# DATA GENERATORS
# ===============================

def get_generators():
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    test_gen = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    return train_gen, test_gen


# ===============================
# MODELS
# ===============================

def build_agroitlite(num_classes):
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.Conv2D(16, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def build_model(base_model, num_classes):
    base_model.trainable = False
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


# ===============================
# TRAIN + EVALUATE
# ===============================

def train_and_evaluate(name, model, train_gen, test_gen):
    print(f"\nTraining {name}...")
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_gen, epochs=EPOCHS, verbose=1)

    print(f"Evaluating {name}...")
    start = time.time()
    predictions = model.predict(test_gen)
    end = time.time()

    y_true = test_gen.classes
    y_pred = np.argmax(predictions, axis=1)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    return {
        "Model": name,
        "Parameters": model.count_params(),
        "Accuracy (%)": round(acc*100, 2),
        "Precision": round(prec,4),
        "Recall": round(rec,4),
        "F1 Score": round(f1,4),
        "Inference Time (s)": round(end-start, 4)
    }


# ===============================
# MAIN PIPELINE
# ===============================

if __name__ == "__main__":

    split_dataset()

    train_gen, test_gen = get_generators()
    num_classes = len(train_gen.class_indices)

    results = []

    # AgroitLite
    model1 = build_agroitlite(num_classes)
    results.append(train_and_evaluate("AgroitLite", model1, train_gen, test_gen))

    # MobileNetV2
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    model2 = build_model(base, num_classes)
    results.append(train_and_evaluate("MobileNetV2", model2, train_gen, test_gen))

    # MobileNetV3
    base = tf.keras.applications.MobileNetV3Small(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    model3 = build_model(base, num_classes)
    results.append(train_and_evaluate("MobileNetV3Small", model3, train_gen, test_gen))

    # EfficientNetB0
    base = tf.keras.applications.EfficientNetB0(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    model4 = build_model(base, num_classes)
    results.append(train_and_evaluate("EfficientNetB0", model4, train_gen, test_gen))

    df = pd.DataFrame(results)
    df = df.sort_values(by="Accuracy (%)", ascending=False)

    print("\n===== FINAL COMPARISON TABLE =====")
    print(df)

    df.to_csv("final_model_comparison.csv", index=False)

    print("\nResults saved as final_model_comparison.csv")
