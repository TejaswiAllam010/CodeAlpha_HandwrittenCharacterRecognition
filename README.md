Handwritten Digit Recognition using CNN

Project Overview

This project is developed as part of the CodeAlpha Machine Learning Internship.

The Handwritten Digit Recognition system uses a Convolutional Neural Network (CNN) to recognize handwritten digits from 0 to 9.

The model is trained using the MNIST dataset and predicts digits from input images.

Technologies Used

- Python
- TensorFlow and Keras
- NumPy
- Matplotlib
- Scikit-learn
- OpenCV
- Pillow

Dataset

- Dataset: MNIST
- Training Images: 60,000
- Testing Images: 10,000
- Image Size: 28 × 28 pixels
- Classes: Digits 0–9

Model Architecture

1. Input Layer
2. Convolutional Layer (32 filters)
3. Max Pooling Layer
4. Convolutional Layer (64 filters)
5. Max Pooling Layer
6. Flatten Layer
7. Dense Layer (128 neurons)
8. Dropout Layer
9. Output Layer (10 classes)

Model Performance

- Algorithm: Convolutional Neural Network
- Training Epochs: 5
- Test Accuracy: 99.03%

Project Structure

CodeAlpha_HandwrittenCharacterRecognition/
│
├── data/
│   └── digit.png
├── models/
│   └── handwritten_digit_model.keras
├── outputs/
│   ├── accuracy_graph.png
│   ├── classification_report.txt
│   ├── confusion_matrix.txt
│   └── confusion_matrix.png
├── src/
│   ├── train_model.py
│   └── predict_digit.py
├── README.md
└── requirements.txt

How to Run

1. Install dependencies

pip install -r requirements.txt

2. Train the model

python src/train_model.py

3. Predict a handwritten digit

python src/predict_digit.py

Enter the image path when prompted, for example:

data/digit.png

Results

The CNN model achieved 99.03% test accuracy on the MNIST test dataset.

The project also generates an accuracy graph, classification report, and confusion matrix.

Internship

Developed as part of the CodeAlpha Machine Learning Internship.