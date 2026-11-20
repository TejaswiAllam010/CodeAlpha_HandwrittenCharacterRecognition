import cv2
import numpy as np
import tensorflow as tf
import os

# Load trained CNN model
model = tf.keras.models.load_model(
    "models/handwritten_digit_model.keras"
)

# Image path
image_path = input("Enter handwritten digit image path: ")

# Check image
if not os.path.exists(image_path):
    print("Image not found! Please check the image path.")
else:
    # Read image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Unable to read image.")
    else:
        # Resize image to 28x28
        image = cv2.resize(image, (28, 28))

        # Invert image if background is white
        if np.mean(image) > 127:
            image = 255 - image

        # Normalize image
        image = image.astype("float32") / 255.0

        # Reshape image
        image = image.reshape(1, 28, 28, 1)

        # Predict digit
        prediction = model.predict(image, verbose=0)
        predicted_digit = np.argmax(prediction)

        confidence = np.max(prediction) * 100

        print("\nPredicted Digit:", predicted_digit)
        print("Confidence:", round(confidence, 2), "%")