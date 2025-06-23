HEAD
import cv2
import mediapipe as mp
import os
import numpy as np
import string
from datetime import datetime

# Create 'data' folder if it doesn't exist
DATA_DIR = 'data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Webcam feed
cap = cv2.VideoCapture(0)

print("[INFO] Press any A–Z key to save sample. Press '27' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Flip and convert color
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)
    landmarks = []

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])  # 63 features
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display webcam window
    cv2.imshow("Sign Language Data Collection", frame)

   
    # Save data when A-Z is pressed
    key = cv2.waitKey(10)
    #ESC key to quit (key code 27)
    if key == 27:
        break

    # Save data when A-Z is pressed
    elif key in range(ord('a'), ord('z') + 1) and landmarks:
        label = chr(key).upper()
        sample = np.array(landmarks)
        label_dir = os.path.join(DATA_DIR, label)
        os.makedirs(label_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
        np.save(os.path.join(label_dir, f'{timestamp}.npy'), sample)
        print(f"[SAVED] Label: {label}, Samples in {label_dir}: {len(os.listdir(label_dir))}")


 

# Cleanup
cap.release()
cv2.destroyAllWindows()

import cv2
import mediapipe as mp
import os
import numpy as np
import string
from datetime import datetime

# Create 'data' folder if it doesn't exist
DATA_DIR = 'data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Webcam feed
cap = cv2.VideoCapture(0)

print("[INFO] Press any A–Z key to save sample. Press '27' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Flip and convert color
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)
    landmarks = []

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])  # 63 features
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display webcam window
    cv2.imshow("Sign Language Data Collection", frame)

   
    # Save data when A-Z is pressed
    key = cv2.waitKey(10)
    #ESC key to quit (key code 27)
    if key == 27:
        break

    # Save data when A-Z is pressed
    elif key in range(ord('a'), ord('z') + 1) and landmarks:
        label = chr(key).upper()
        sample = np.array(landmarks)
        label_dir = os.path.join(DATA_DIR, label)
        os.makedirs(label_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
        np.save(os.path.join(label_dir, f'{timestamp}.npy'), sample)
        print(f"[SAVED] Label: {label}, Samples in {label_dir}: {len(os.listdir(label_dir))}")


 

# Cleanup
cap.release()
cv2.destroyAllWindows()
