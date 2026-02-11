import os
import random
import shutil
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# =========================================
# REPRODUCIBILITY
# =========================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# =========================================
# CONFIG
# =========================================
ORIGINAL_DATASET = "dataset"
SPLIT_DATASET = "dataset_split"
TRAIN_DIR = os.path.join(SPLIT_DATASET, "train")
TEST_DIR = os.path.join(SPLIT_DATASET, "test")

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 25

# =========================================
# STEP 1 — TRAIN/TEST SPLIT
# =========================================
def split_dataset(train_ratio=0.8):

    if os.path.exists(SPLIT_DATASET):
        print("Dataset already split.")
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
            shutil.copy(
                os.path.join(class_path, img),
                os.path.join(TRAIN_DIR, class_name, img)
            )

        for img in test_images:
            shutil.copy(
                os.path.join(class_path, img),
                os.path.join(TEST_DIR, class_name, img)
            )

    print("Dataset split complete.")

# =========================================
# DATA GENERATORS
# =========================================
def get_generators():

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        seed=SEED
    )

    test_gen = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    return train_gen, test_gen

# =========================================
# AGROITLITE MODEL (MATCHES DIAGRAM)
# =========================================
def build_agroitlite(num_classes):

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    # Initial Conv
    x = layers.Conv2D(16, (3,3), padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Depthwise Separable Block 1
    x = layers.DepthwiseConv2D((3,3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(32, (1,1), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Depthwise Separable Block 2
    x = layers.DepthwiseConv2D((3,3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(64, (1,1), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Global Avg Pool
    x = layers.GlobalAveragePooling2D()(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)

    return model

# =========================================
# MAIN
# =========================================
if __name__ == "__main__":

    split_dataset()

    train_gen, test_gen = get_generators()
    num_classes = len(train_gen.class_indices)

    model = build_agroitlite(num_classes)

    print("\nModel Summary:\n")
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=3,
        min_lr=1e-6
    )

    print("\nTraining AgroitLite...\n")

    history = model.fit(
        train_gen,
        validation_data=test_gen,
        epochs=EPOCHS,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    # Save full model
    model.save("AgroitLite_final_model.h5")
    print("\nModel saved as AgroitLite_final_model.h5")

    # =========================================
    # EVALUATION
    # =========================================

    predictions = model.predict(test_gen)
    y_true = test_gen.classes
    y_pred = np.argmax(predictions, axis=1)

    class_labels = list(test_gen.class_indices.keys())

    acc = np.mean(y_true == y_pred)
    print(f"\nFinal Test
