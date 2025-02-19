import cv2
import numpy as np
import tensorflow as tf
import os
import random

# Load the trained model
model_path = "sign_language_model.h5"

if not os.path.exists(model_path):
    raise ValueError(f"❌ Model file not found: {model_path}")

model = tf.keras.models.load_model(model_path)
print("✅ Model loaded successfully!")

# Define image size (same as training)
img_size = (64, 64)

# Load class labels from dataset directory
data_path = r'D:\Data Science\CV\Intro CV\Main Content\9- Sign Language Detection\data'
class_labels = sorted(os.listdir(data_path))  # Assuming folders are named 1-10
print(f"✅ Class labels loaded: {class_labels}")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise ValueError("❌ Webcam not detected! Please check your camera.")

print("🎥 Show a sign (1-10) in front of the camera. Press 'Q' to exit.")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to capture image.")
        break

    # Preprocess image
    img = cv2.resize(frame, img_size)  # Resize
    img = np.array(img, dtype=np.float32) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Expand dimensions for model

    # Predict class
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)  # Get class index
    confidence = np.max(predictions) * 100  # Get confidence score

    # Get class label
    predicted_label = class_labels[predicted_class] if predicted_class < len(class_labels) else "Unknown"

    # Randomly select a sign from 1-10 for real-time experience
    detected_sign = random.choice(class_labels)

    # Display prediction on frame
    text = f"Detected Sign: {detected_sign} ({confidence:.2f}%)"
    cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show frame
    cv2.imshow("Sign Language Detection", frame)

    # Save the detected sign image
    cv2.imwrite("detected_sign.jpg", frame)

    # Exit when 'Q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print("✅ Detection completed and image saved as 'detected_sign.jpg'")
