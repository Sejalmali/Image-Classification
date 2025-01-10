
# Image Classification of Handwritten Digits (MNIST)

This repository contains code to classify images of handwritten digits (0-9) using a Convolutional Neural Network (CNN). The project demonstrates training, evaluation, and prediction capabilities using the MNIST dataset.

---

## Problem Statement
Classify images of handwritten digits into categories ranging from 0 to 9 using deep learning techniques.

---

## Author
**Sejal Anil Mali**  

---

## Dataset
The dataset used in this project is the MNIST dataset, which is preloaded in the `keras.datasets` module. It consists of:
- **Training Data**: 60,000 images
- **Test Data**: 10,000 images  
Each image is grayscale with a resolution of 28x28 pixels.

---

## Project Overview
### Key Features:
1. **Data Visualization**: Display of sample training images with their corresponding labels.
2. **Model Architecture**:
   - Input layer: Convolutional layers with ReLU activation.
   - Pooling layers for dimensionality reduction.
   - Dense layers leading to a final softmax output layer for classification.
3. **Model Training**: 
   - Optimizer: Adam
   - Loss function: Categorical Crossentropy
   - Metric: Accuracy
4. **Evaluation**: Model evaluation on test data with accuracy reporting.
5. **Prediction**: Predicting the class of a single handwritten digit image.

---

## Requirements
- Python 3.x
- TensorFlow/Keras
- NumPy
- Matplotlib

Install the required libraries using:
```bash
pip install tensorflow numpy matplotlib
