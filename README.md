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
   ```bash pip install -r requirements.txt
  
3. Run the Webcam Application
   To start real-time sign language detection using OpenCV:
   ```bash python opencv.py

# Requirements
  numpy==1.26.4 
  
  opencv-python==4.11.0.86
  
  matplotlib==3.9.2
  
  mediapipe==0.10.21
  
  tensorflow==2.19.0

# Supported Signs
This project supports static ASL letters A-Y (excluding J and Z due to motion).



