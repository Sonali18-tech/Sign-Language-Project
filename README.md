🧠 Real-Time Sign Language Recognition with Voice 🔊
<br>
This project is a Streamlit-based real-time sign language recognition web app that recognizes static hand gestures (A–Z) using MediaPipe, classifies them using a trained ML model, constructs full sentences, and optionally reads them aloud using Text-to-Speech (gTTS).

<br>
🚀 Features
✅ Real-time hand tracking using MediaPipe
✅ Letter-by-letter prediction using a trained ML model
✅ Automatic sentence construction with autocorrect (optional)
✅ Voice output via Google Text-to-Speech
✅ Clean, interactive Streamlit UI with camera control
✅ Model compression using base64 in compress_model.py (no extra .pkl file needed)
                 
⚙️ Requirements
This app runs on:

Python 3.10

Streamlit ≥ 1.25

OpenCV

MediaPipe 0.10.21

gTTS

TextBlob

NumPy
📦 Installation
1. Clone the repository:
   git clone https://github.com/Sonali18-tech/Sign-Language-Project.git
cd Sign-Language-Project
2. Create and activate virtual environment:
   conda create -n signlang python=3.10 -y
conda activate signlang
3. Install dependencies:
   pip install -r requirements.txt
4. Run the Streamlit app:
   streamlit run streamlit_app.py
📍 Open your browser and go to: http://localhost:8501

📹 How to Use
Launch the app

Click ✅ "Start Camera"

Show hand signs (A–Z) to your webcam

Watch real-time letter prediction and sentence construction

Press "🔊 Speak Now" to convert sentence to speech

Click "❌ Clear Sentence" to reset

🧪 Model Details
Input: 63 features from MediaPipe hand landmarks

Classifier: Trained with scikit-learn

Data: Custom hand gesture dataset collected using MediaPipe

🔐 Troubleshooting
Camera not working?

Close other apps using your webcam (Zoom, Teams, etc.)

Try running in another browser

App not loading on Streamlit Cloud?

Make sure mediapipe==0.10.21 is compatible with Python 3.10

Avoid Python 3.11+ on Streamlit Cloud (use 3.10)

🙌 Credits
Built by Sonali Paliwal during AIML Internship

Powered by MediaPipe, Streamlit, and gTTS














