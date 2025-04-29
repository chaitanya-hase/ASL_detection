import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model(r"my_model.keras")

# Define constants
IMG_SIZE = 64  # Should match your model input size
class_labels = list(model.classes) if hasattr(model, 'classes') else [chr(i) for i in range(65, 65+model.output_shape[-1])]

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)
with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get bounding box of hand
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]
                xmin = int(min(x_coords) * w) - 20
                ymin = int(min(y_coords) * h) - 20
                xmax = int(max(x_coords) * w) + 20
                ymax = int(max(y_coords) * h) + 20

                # Crop hand region
                roi = frame[ymin:ymax, xmin:xmax]
                if roi.size == 0:
                    continue
                roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
                roi = roi.astype("float32") / 255.0
                roi = np.expand_dims(roi, axis=0)

                # Predict
                pred = model.predict(roi)
                class_index = np.argmax(pred)
                confidence = np.max(pred)
                predicted_label = class_labels[class_index]

                # Display
                cv2.putText(frame, f"{predicted_label} ({confidence*100:.2f}%)", 
                            (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255,0,0), 2)
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("ASL Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            break

cap.release()
cv2.destroyAllWindows()
