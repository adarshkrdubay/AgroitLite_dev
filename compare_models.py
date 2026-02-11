import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ==============================
# CONFIG
# ==============================

TEST_DIR = "dataset/test"  # Change if needed
IMG_SIZE = 224
BATCH_SIZE = 32

MODEL_PATHS = {
    "AgroitLite": "Models/agroitlite_mango.h5",
    "MobileNetV2": "mobileNetV2_model.h5",
    "ResNet50": "resnet50_model.h5",
    "MobileNetV3": "mobilenetv3_model.h5",
    "EfficientNet": "efficientnet_model.h5"
}

# ==============================
# LOAD TEST DATA
# ==============================

datagen = ImageDataGenerator(rescale=1./255)

test_generator = datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

y_true = test_generator.classes
class_indices = test_generator.class_indices
num_classes = len(class_indices)

# ==============================
# FUNCTIONS
# ==============================

def get_model_size_mb(model_path):
    return round(os.path.getsize(model_path) / (1024 * 1024), 2)

def evaluate_model(name, model_path):
    print(f"\nEvaluating {name}...")

    model = tf.keras.models.load_model(model_path)

    # Parameter count
    params = model.count_params()

    # Inference timing
    start_time = time.time()
    predictions = model.predict(test_generator, verbose=0)
    end_time = time.time()

    inference_time = end_time - start_time
    avg_latency = inference_time / len(y_true)

    y_pred = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    model_size = get_model_size_mb(model_path)

    return {
        "Model": name,
        "Parameters": params,
        "Size (MB)": model_size,
        "Accuracy (%)": round(accuracy * 100, 2),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1 Score": round(f1, 4),
        "Total Inference Time (s)": round(inference_time, 4),
        "Avg Latency per Image (ms)": round(avg_latency * 1000, 4)
    }

# ==============================
# MAIN EVALUATION LOOP
# ==============================

results = []

for model_name, model_path in MODEL_PATHS.items():
    if os.path.exists(model_path):
        result = evaluate_model(model_name, model_path)
        results.append(result)
    else:
        print(f"Model not found: {model_path}")

# ==============================
# CREATE TABLE
# ==============================

df = pd.DataFrame(results)
df = df.sort_values(by="Accuracy (%)", ascending=False)

print("\n===== COMPARISON TABLE =====")
print(df)

df.to_csv("model_comparison_results.csv", index=False)

print("\nTable saved as model_comparison_results.csv")
