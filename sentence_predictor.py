# Add at the top
import cv2
import mediapipe as mp
import numpy as np
import pickle
import pyttsx3
import time

# Load model
with open('sign_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Init
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
engine = pyttsx3.init()
engine.setProperty('rate', 150)

sentence = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    
    if result.multi_hand_landmarks:
       for hand_landmarks in result.multi_hand_landmarks:
           landmarks = []
           for lm in hand_landmarks.landmark:
               landmarks.extend([lm.x, lm.y, lm.z])

           # DEBUG: Print feature length
           print(f"Len(landmarks): {len(landmarks)}")

           if len(landmarks) == 63:  # Match your model training
               prediction = model.predict([landmarks])[0]
               confidence = max(model.predict_proba([landmarks])[0])
               print(f"Prediction: {prediction}, Confidence: {confidence}")

               if confidence > 0.4:  # lower for now
                   if len(sentence) == 0 or sentence[-1] != prediction:
                       sentence += prediction
                   cv2.putText(frame, f"Prediction: {prediction}", (10, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

           mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display full sentence
    cv2.putText(frame, f"Sentence: {sentence}", (10, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Sign Language to Sentence", frame)

    key = cv2.waitKey(1)

    if key == ord(' '):  # Speak sentence
        if sentence:
            print(f"Speaking: {sentence}")
            engine.say(sentence)
            engine.runAndWait()
            sentence = ""
            time.sleep(1)

    if key == ord('q'):  # Quit
        break

cap.release()
cv2.destroyAllWindows()
