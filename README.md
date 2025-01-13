# ASL Sign Language Detection Using CNN
Welcome to the ASL Sign Language Detection Using CNN project! This repository contains a Convolutional Neural Network (CNN) model designed to recognize American Sign Language (ASL) gestures in real-time through webcam input.

---
🌟  **Project Description :** 
The ASL Sign Language Detection Using CNN project aims to bridge communication gaps by recognizing ASL gestures using computer vision and deep learning. This project uses a trained CNN model to classify ASL signs into different categories and display the results in real-time.

---
🚀 **Features**

**Real-Time ASL Gesture Recognition:** Capture hand gestures through a webcam and classify them in real-time.

**Pre-Trained CNN Model:** A robust CNN model trained on ASL sign data.

**User-Friendly Interface:** Simple and intuitive user experience with real-time feedback.

**High Accuracy:** Achieve high accuracy with optimized model architecture.

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

## 📥 Installation and Set-up
To get started with the project, follow these steps:
1. Clone the repository:
    ```bash
    git clone https://github.com/Abraham-Samuel470/ASL_SignLanguageDetectionUsingCNN.git
cd ASL_SignLanguageDetectionUsingCNN

    ```
2. Interact with the System:
   
    -Show different ASL signs in front of your webcam.
   
    -The system will recognize the gesture and display the corresponding letter or action.

4. Exit the Application:
   
   -Press `ESC` or close the window to stop the detection.
   
5. Install the required dependencies:
    ```bash
    pip install tensorflow keras numpy matplotlib opencv-python scikit-learn
    ```
6. Download or prepare your ASL dataset and place it in the `data` directory or create your own data directory like how i create mine as `ASLdatesheet`.

---

## 🎮 Usage
To run the ASL Sign Language Detection:

1. Start the Detection:
    ```bash
    python livedetection.py

    ```

2. Interact with the System:
   
    -Show different ASL signs in front of your webcam.
   
    -The system will recognize the gesture and display the corresponding letter or action.

3. Exit the Application:
   
   -Press `ESC` or close the window to stop the detection.

---

🛠 **Contributing** 
We welcome contributions to enhance the project! Here's how you can contribute:
1.Fork the repository.

2.Create a new branch for your feature or bug fix.

3.Commit your changes and push them to your fork.

4.Open a pull request with a detailed description of your changes.

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
