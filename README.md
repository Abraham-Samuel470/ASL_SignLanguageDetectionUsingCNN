# ASL Sign Language Detection Using CNN

A machine learning project for detecting American Sign Language (ASL) gestures using a Convolutional Neural Network (CNN). This project processes hand gesture images and classifies them into corresponding ASL letters or words.

---

## Features
- Trains a CNN model on ASL dataset images.
- Predicts ASL letters or words in real-time or from static images.
- Supports customization for additional gestures or extended datasets.

---

## Dataset
The project uses an ASL dataset containing images of hand gestures for each letter of the alphabet. You can use publicly available datasets or create your own dataset.

Suggested Dataset Sources:
- [Kaggle ASL Dataset](https://www.kaggle.com/)
- [Sign Language MNIST](https://www.kaggle.com/datamunge/sign-language-mnist)

---

## Modules Used
| Module               | Description                                                                | Link to Install                                                              |
|----------------------|----------------------------------------------------------------------------|------------------------------------------------------------------------------|
| `tensorflow`         | Framework for building and training the CNN model.                        | [TensorFlow](https://pypi.org/project/tensorflow/)                           |
| `keras`              | High-level API for TensorFlow to build neural networks.                   | [Keras](https://pypi.org/project/keras/)                                    |
| `numpy`              | Library for numerical operations on data.                                | [NumPy](https://pypi.org/project/numpy/)                                    |
| `matplotlib`         | Visualization library for plotting training metrics and results.          | [Matplotlib](https://pypi.org/project/matplotlib/)                          |
| `opencv-python`      | Library for real-time image processing and webcam feed handling.          | [OpenCV](https://pypi.org/project/opencv-python/)                           |
| `sklearn`            | For preprocessing, splitting data, and evaluating model performance.      | [scikit-learn](https://pypi.org/project/scikit-learn/)                      |

---

## Setup and Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/ASL_SignLanguageDetectionUsingCNN.git
    ```
2. Navigate to the project directory:
    ```bash
    cd ASL_SignLanguageDetectionUsingCNN
    ```
3. Install the required dependencies:
    ```bash
    pip install tensorflow keras numpy matplotlib opencv-python scikit-learn
    ```
4. Download or prepare your ASL dataset and place it in the `data` directory.

---

## Usage
1. Train the CNN model:
    ```bash
    python train_model.py
    ```
    - This script loads the dataset, preprocesses it, and trains a CNN model.
    - Model weights are saved in the `model/` directory.

2. Test the model with static images:
    ```bash
    python test_model.py --image path/to/image.jpg
    ```

3. Predict gestures in real-time:
    ```bash
    python predict_realtime.py
    ```
    - Opens a webcam feed and predicts ASL gestures in real-time.

---

## Directory Structure
```plaintext
ASL_SignLanguageDetectionUsingCNN/
│
├── data/                     # Dataset folder
├── model/                    # Trained model weights
├── train_model.py            # Script to train the CNN model
├── test_model.py             # Script to test the model on static images
├── predict_realtime.py       # Script for real-time gesture detection
├── utils/                    # Utility scripts for preprocessing, etc.
├── README.md                 # Project documentation
└── requirements.txt          # List of required Python packages(tenserflow,etc)
