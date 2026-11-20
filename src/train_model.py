import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix

# Create required folders
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print("Training samples:", len(x_train))
print("Testing samples:", len(x_test))

# Normalize images
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Add channel dimension
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# Build CNN model
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(10, activation="softmax")
])

# Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train model
history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1
)

# Evaluate model
test_loss, test_accuracy = model.evaluate(x_test, y_test)

print("\nTest Accuracy:", round(test_accuracy * 100, 2), "%")

# Save trained model
model.save("models/handwritten_digit_model.keras")

# Save training accuracy graph
plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("CNN Model Accuracy")
plt.legend()
plt.savefig("outputs/accuracy_graph.png")
plt.close()

# Generate predictions
predictions = model.predict(x_test)
predicted_labels = np.argmax(predictions, axis=1)

# Save classification report
report = classification_report(y_test, predicted_labels)
with open("outputs/classification_report.txt", "w") as file:
    file.write(report)

# Save confusion matrix
cm = confusion_matrix(y_test, predicted_labels)

plt.figure(figsize=(10, 8))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.colorbar()

for i in range(10):
    for j in range(10):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.xticks(range(10))
plt.yticks(range(10))
plt.savefig("outputs/confusion_matrix.png")
plt.close()

np.savetxt("outputs/confusion_matrix.txt", cm, fmt="%d")

print("\nModel saved successfully!")
print("Accuracy graph and evaluation reports saved in outputs folder.")