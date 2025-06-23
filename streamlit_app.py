HEAD
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle
from gtts import gTTS
import tempfile
import os
import time
from textblob import TextBlob
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'sign_model.pkl')
MAX_FRAME_WIDTH = 640  # Reduced resolution for better performance

@st.cache_resource
def load_model():
    """Load and cache the sign language model"""
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        st.error("❌ Failed to load sign language model")
        st.stop()

@st.cache_resource
def get_hands_processor():
    """Initialize and cache MediaPipe hands processor"""
    mp_hands = mp.solutions.hands
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

def generate_audio(text):
    """Generate and return audio file path for given text"""
    try:
        tts = gTTS(text=text)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_file.name)
        return temp_file.name
    except Exception as e:
        logger.error(f"Audio generation failed: {str(e)}")
        st.error("❌ Failed to generate speech")
        return None

def main():
    # Load resources
    model = load_model()
    hands = get_hands_processor()
    mp_drawing = mp.solutions.drawing_utils

    # Streamlit UI
    st.title("🧠 Real-Time Sign Language Recognition with Voice 🔊")
    
    with st.expander("ℹ️ How to use"):
        st.write("""
        1. Click 'Start Camera' to begin
        2. Show hand signs to the camera
        3. Use 'Speak Now' to hear the sentence
        4. 'Clear Sentence' to start over
        """)

    col1, col2 = st.columns(2)
    start = col1.checkbox("📷 Start Camera")
    stop = col2.button("🛑 Stop")

    autocorrect_on = st.checkbox("✅ Enable Autocorrect", value=True)

    # Initialize session state
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

    # UI Elements
    FRAME_WINDOW = st.image([], use_column_width=True)
    pred_text = st.empty()
    sentence_display = st.empty()

    col3, col4 = st.columns(2)
    speak_btn = col3.button("🔊 Speak Now")
    clear_btn = col4.button("❌ Clear Sentence")

    # Camera processing
    if st.session_state.camera_on:
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("❌ Could not access camera. Try refreshing the page.")
                st.session_state.camera_on = False
                st.stop()

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, MAX_FRAME_WIDTH)
            
            frame_placeholder = st.empty()
            with st.spinner("Processing camera feed..."):
                frame_count = 0
                while st.session_state.camera_on:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("❌ Camera feed interrupted")
                        break

                    # Process frame
                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(rgb)

                    frame_count += 1
                    if results.multi_hand_landmarks and frame_count % 10 == 0:
                        for hand_landmarks in results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                            
                            landmarks = []
                            for lm in hand_landmarks.landmark:
                                landmarks.extend([lm.x, lm.y, lm.z])

                            if len(landmarks) == 63:
                                try:
                                    prediction = model.predict([landmarks])[0]
                                    if prediction != st.session_state.prev_letter:
                                        st.session_state.sentence += prediction
                                        st.session_state.prev_letter = prediction
                                    pred_text.markdown(f"### 🔠 Predicted Letter: `{prediction}`")
                                except Exception as e:
                                    logger.error(f"Prediction failed: {str(e)}")
                                    continue

                    # Update sentence with autocorrect if enabled
                    if autocorrect_on and st.session_state.sentence:
                        try:
                            st.session_state.constructed_sentence = str(TextBlob(st.session_state.sentence).correct())
                        except:
                            st.session_state.constructed_sentence = st.session_state.sentence
                    else:
                        st.session_state.constructed_sentence = st.session_state.sentence

                    # Update UI
                    sentence_display.text_area(
                        "📝 Constructed Sentence",
                        st.session_state.constructed_sentence,
                        height=100,
                        key=f"sentence_display_{frame_count}"
                    )

                    FRAME_WINDOW.image(frame, channels="BGR")

                    if stop:
                        break

                    time.sleep(0.03)

        except Exception as e:
            logger.error(f"Camera error: {str(e)}")
            st.error("❌ Camera processing failed")
        finally:
            if 'cap' in locals():
                cap.release()
            st.session_state.camera_on = False
    else:
        st.info("☝️ Tick 'Start Camera' to begin")

    # Handle button actions
    if clear_btn:
        st.session_state.sentence = ""
        st.session_state.prev_letter = ""
        st.session_state.constructed_sentence = ""
        st.rerun()

    if speak_btn and st.session_state.constructed_sentence.strip():
        with st.spinner("Generating speech..."):
            audio_file = generate_audio(st.session_state.constructed_sentence)
            if audio_file:
                st.success(f"✅ Speaking: {st.session_state.constructed_sentence}")
                st.audio(audio_file, format="audio/mp3")
                try:
                    os.unlink(audio_file)  # Clean up audio file
                except:
                    pass

if __name__ == "__main__":
    main()

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle
from gtts import gTTS
import tempfile
import os
import time
from textblob import TextBlob
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'sign_model.pkl')
MAX_FRAME_WIDTH = 640  # Reduced resolution for better performance

@st.cache_resource
def load_model():
    """Load and cache the sign language model"""
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        st.error("❌ Failed to load sign language model")
        st.stop()

@st.cache_resource
def get_hands_processor():
    """Initialize and cache MediaPipe hands processor"""
    mp_hands = mp.solutions.hands
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

def generate_audio(text):
    """Generate and return audio file path for given text"""
    try:
        tts = gTTS(text=text)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_file.name)
        return temp_file.name
    except Exception as e:
        logger.error(f"Audio generation failed: {str(e)}")
        st.error("❌ Failed to generate speech")
        return None

def main():
    # Load resources
    model = load_model()
    hands = get_hands_processor()
    mp_drawing = mp.solutions.drawing_utils

    # Streamlit UI
    st.title("🧠 Real-Time Sign Language Recognition with Voice 🔊")
    
    with st.expander("ℹ️ How to use"):
        st.write("""
        1. Click 'Start Camera' to begin
        2. Show hand signs to the camera
        3. Use 'Speak Now' to hear the sentence
        4. 'Clear Sentence' to start over
        """)

    col1, col2 = st.columns(2)
    start = col1.checkbox("📷 Start Camera")
    stop = col2.button("🛑 Stop")

    autocorrect_on = st.checkbox("✅ Enable Autocorrect", value=True)

    # Initialize session state
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

    # UI Elements
    FRAME_WINDOW = st.image([], use_column_width=True)
    pred_text = st.empty()
    sentence_display = st.empty()

    col3, col4 = st.columns(2)
    speak_btn = col3.button("🔊 Speak Now")
    clear_btn = col4.button("❌ Clear Sentence")

    # Camera processing
    if st.session_state.camera_on:
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("❌ Could not access camera. Try refreshing the page.")
                st.session_state.camera_on = False
                st.stop()

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, MAX_FRAME_WIDTH)
            
            frame_placeholder = st.empty()
            with st.spinner("Processing camera feed..."):
                frame_count = 0
                while st.session_state.camera_on:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("❌ Camera feed interrupted")
                        break

                    # Process frame
                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(rgb)

                    frame_count += 1
                    if results.multi_hand_landmarks and frame_count % 10 == 0:
                        for hand_landmarks in results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                            
                            landmarks = []
                            for lm in hand_landmarks.landmark:
                                landmarks.extend([lm.x, lm.y, lm.z])

                            if len(landmarks) == 63:
                                try:
                                    prediction = model.predict([landmarks])[0]
                                    if prediction != st.session_state.prev_letter:
                                        st.session_state.sentence += prediction
                                        st.session_state.prev_letter = prediction
                                    pred_text.markdown(f"### 🔠 Predicted Letter: `{prediction}`")
                                except Exception as e:
                                    logger.error(f"Prediction failed: {str(e)}")
                                    continue

                    # Update sentence with autocorrect if enabled
                    if autocorrect_on and st.session_state.sentence:
                        try:
                            st.session_state.constructed_sentence = str(TextBlob(st.session_state.sentence).correct())
                        except:
                            st.session_state.constructed_sentence = st.session_state.sentence
                    else:
                        st.session_state.constructed_sentence = st.session_state.sentence

                    # Update UI
                    sentence_display.text_area(
                        "📝 Constructed Sentence",
                        st.session_state.constructed_sentence,
                        height=100,
                        key=f"sentence_display_{frame_count}"
                    )

                    FRAME_WINDOW.image(frame, channels="BGR")

                    if stop:
                        break

                    time.sleep(0.03)

        except Exception as e:
            logger.error(f"Camera error: {str(e)}")
            st.error("❌ Camera processing failed")
        finally:
            if 'cap' in locals():
                cap.release()
            st.session_state.camera_on = False
    else:
        st.info("☝️ Tick 'Start Camera' to begin")

    # Handle button actions
    if clear_btn:
        st.session_state.sentence = ""
        st.session_state.prev_letter = ""
        st.session_state.constructed_sentence = ""
        st.rerun()

    if speak_btn and st.session_state.constructed_sentence.strip():
        with st.spinner("Generating speech..."):
            audio_file = generate_audio(st.session_state.constructed_sentence)
            if audio_file:
                st.success(f"✅ Speaking: {st.session_state.constructed_sentence}")
                st.audio(audio_file, format="audio/mp3")
                try:
                    os.unlink(audio_file)  # Clean up audio file
                except:
                    pass

if __name__ == "__main__":
    main()
 
