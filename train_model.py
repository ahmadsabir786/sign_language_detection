import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Path to dataset folder
data_path = r'D:\Data Science\CV\Intro CV\Main Content\9- Sign Language Detection\data'

# Parameters
img_size = (64, 64)  # Resize images
batch_size = 32
epochs = 10

# Data Augmentation & Loading
data_gen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)

train_data = data_gen.flow_from_directory(data_path, target_size=img_size, batch_size=batch_size, subset="training")
val_data = data_gen.flow_from_directory(data_path, target_size=img_size, batch_size=batch_size, subset="validation")

# Get the number of classes (digits 1-10)
num_classes = len(train_data.class_indices)

# Define CNN Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')  # Softmax for multi-class classification
])

# Compile Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train Model
history = model.fit(train_data, validation_data=val_data, epochs=epochs)

# Save Model
model.save("sign_language_model.h5")

# Plot Training Performance
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.show()

print("✅ Model training complete and saved as 'sign_language_model.h5'")
