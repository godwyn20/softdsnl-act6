import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import json
import os

# Paths
train_dir = "data/train"
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)

# Image generators
train_datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode="categorical",
    subset="training"
)
val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode="categorical",
    subset="validation"
)

# Build CNN
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation="relu", input_shape=(64,64,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(len(train_generator.class_indices), activation="softmax")
])

# Compile
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train
history = model.fit(
    train_generator,
    epochs=5,
    validation_data=val_generator
)

# Save model and class names
model_path = os.path.join(models_dir, "kaggle_cnn_model.h5")
json_path = os.path.join(models_dir, "class_names.json")

model.save(model_path)
with open(json_path, "w") as f:
    json.dump(list(train_generator.class_indices.keys()), f)

print(f"✅ Model saved at {model_path}")
print(f"✅ Class names saved at {json_path}")
