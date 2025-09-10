import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import json
import os

# Paths
models_dir = "models"
model_path = os.path.join(models_dir, "kaggle_cnn_model.h5")
json_path = os.path.join(models_dir, "class_names.json")

# Load model
model = tf.keras.models.load_model(model_path)

# Load class names
with open(json_path, "r") as f:
    class_names = json.load(f)

def predict_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(64, 64))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions)]
        confidence = float(np.max(predictions))

        print(f"✅ Prediction: {predicted_class} ({confidence:.2f})")
    except Exception as e:
        print(f"❌ Error processing {img_path}: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("⚠️ Usage: python test_model.py <image_path>")
    else:
        img_path = sys.argv[1]
        predict_image(img_path)
