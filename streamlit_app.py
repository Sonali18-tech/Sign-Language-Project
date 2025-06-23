import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle
import pyttsx3
import threading
import queue
import time
from textblob import TextBlob

# Load model
with open('sign_model.pkl', 'rb') as f:
    model = pickle.load(f)

# --- TTS Thread Setup ---
def tts_worker(q):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1.0)
    while True:
        text = q.get()
        if text == "EXIT":
            break
        engine.say(text)
        engine.runAndWait()
        q.task_done()

if 'tts_queue' not in st.session_state:
    st.session_state.tts_queue = queue.Queue()
    threading.Thread(target=tts_worker, args=(st.session_state.tts_queue,), daemon=True).start()

def speak_text(sentence):
    st.session_state.tts_queue.put(sentence)

# Mediapipe Init
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Streamlit UI
st.title("🧠 Real-Time Sign Language Recognition with Voice 🔊")
col1, col2 = st.columns(2)
start = col1.checkbox("📷 Start Camera")
stop = col2.button("🛑 Stop")

autocorrect_on = st.checkbox("✅ Enable Autocorrect", value=True)

FRAME_WINDOW = st.image([])
pred_text = st.empty()
sentence_display = st.empty()

col3, col4 = st.columns(2)
speak_btn = col3.button("🔊 Speak Now")
clear_btn = col4.button("❌ Clear Sentence")

# Session State Setup
if 'camera_on' not in st.session_state:
    st.session_state.camera_on = False
if 'prev_letter' not in st.session_state:
    st.session_state.prev_letter = ""
if 'sentence' not in st.session_state:
    st.session_state.sentence = ""
if 'constructed_sentence' not in st.session_state:
    st.session_state.constructed_sentence = ""

# Start/Stop logic
if start:
    st.session_state.camera_on = True
if stop:
    st.session_state.camera_on = False

# Camera loop
if st.session_state.camera_on:
    cap = cv2.VideoCapture(0)
    frame_count = 0

    while st.session_state.camera_on:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Cannot access webcam")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        frame_count += 1
        if results.multi_hand_landmarks and frame_count % 10 == 0:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = [lm.x for lm in hand_landmarks.landmark]
                landmarks += [lm.y for lm in hand_landmarks.landmark]
                landmarks += [lm.z for lm in hand_landmarks.landmark]

                if len(landmarks) == 63:
                    prediction = model.predict([landmarks])[0]
                    if prediction != st.session_state.prev_letter:
                        st.session_state.sentence += prediction
                        st.session_state.prev_letter = prediction
                    pred_text.markdown(f"### 🔠 Predicted Letter: `{prediction}`")

        # Show sentence with optional autocorrect
        st.session_state.constructed_sentence = str(TextBlob(st.session_state.sentence).correct()) if autocorrect_on else st.session_state.sentence
        
        # Update the sentence display
        sentence_display.text_area(
            "📝 Constructed Sentence", 
            st.session_state.constructed_sentence, 
            height=100, 
            key=f"sentence_display_{frame_count}"
        )

        FRAME_WINDOW.image(frame)

        if stop:
            break

        time.sleep(0.03)

    cap.release()
    st.session_state.camera_on = False
    st.session_state.tts_queue.put("EXIT")

else:
    st.info("☝️ Tick 'Start Camera' to begin")

# Handle button presses outside the camera loop
if clear_btn:
    st.session_state.sentence = ""
    st.session_state.prev_letter = ""
    st.session_state.constructed_sentence = ""
    # Force a rerun to update the display immediately
    st.rerun()

if speak_btn and st.session_state.constructed_sentence.strip():
    st.success(f"✅ Speaking: {st.session_state.constructed_sentence}")
    speak_text(st.session_state.constructed_sentence)