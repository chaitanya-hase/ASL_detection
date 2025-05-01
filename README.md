# 🧠 Sign Language Detection using CNN & MediaPipe

This project uses a Convolutional Neural Network (CNN) model along with MediaPipe hand tracking to detect and classify American Sign Language (ASL) hand gestures in real-time through a webcam.

## 🚀 Features

- Real-time hand gesture detection
- Pre-trained CNN model for ASL alphabets A–Y (excluding J and Z)
- Uses MediaPipe for hand landmark detection
- Smooth predictions with frame averaging
- Built using TensorFlow, OpenCV, and MediaPipe

## 🛠️ Installation

1. Clone the repository
   ```bash
   git clone https://github.com/chaitanya-hase/ASL_detection.git
   cd ASL_detection

2. Install Required Packages
   
   ```bash
   pip install -r requirements.txt
  
4. Run the Webcam Application
   To start real-time sign language detection using OpenCV:

   ```bash
   python opencv.py

# Requirements
  numpy==1.26.4 
  
  opencv-python==4.11.0.86
  
  matplotlib==3.9.2
  
  mediapipe==0.10.21
  
  tensorflow==2.19.0

# Supported Signs
This project supports static ASL letters A-Z

# Proposed Solution
The proposed system aims to bridge the communication gap between hearing-impaired individuals and others by recognizing hand signs using computer vision and deep learning. The solution      includes the following components:
     1. Data Collection:
         Use a labeled dataset of hand sign images.
         Include variations in hand gestures, lighting, and background to improve model          
         generalization.
         
     2. Data Preprocessing:
         Resize, normalize, and augment images to increase dataset diversity.
         
     3. Deep Learning Algorithm:
         Build and train a Convolutional Neural Network (CNN) using TensorFlow/Keras.
         Fine-tune the model with techniques like data augmentation and learning rate scheduling for better accuracy.
         
      4. Real-Time Detection:
         Integrate the trained model with OpenCV to detect hand gestures in real-time using a webcam.
         Display the predicted letter on screen for instant interpretation.
         
      5. Deployment:
         Create a simple user interface or Python script for easy access.
         Optionally deploy as a desktop app or web interface using Flask or Streamlit.
         
      6. Evaluation:
         Measure model accuracy and confusion matrix to evaluate performance.
         Continuously improve based on test results and real-time performance.
         
   
# System  Approach
   Programming Language: Python

  Frameworks: TensorFlow and Keras for building and training the Convolutional Neural Network(CNN)

  Computer Vision: OpenCV for real-time webcam input and Mediapipe for accurate hand tracking

  Dataset Used: American Sign Language (ASL) Alphabet Dataset — containing labeled images of hand gestures representing the English alphabet

  Development Environment: Visual Studio Code on Windows 

  Libraries: NumPy, Matplotlib, Mediapipe, TensorFlow, OpenCV

  
