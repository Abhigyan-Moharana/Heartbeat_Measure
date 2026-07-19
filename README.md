![PulseSense Banner](https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=2,12,20&height=220&section=header&text=PulseSense&fontSize=70&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=See%20Your%20Heart%20Rate,%20Not%20Just%20Feel%20It&descAlignY=58&descColor=ffffff)
![Typing SVG](https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=22&duration=2500&pause=800&color=FF4B4B&center=true&vCenter=true&width=650&lines=No+Wearables.+No+Wires.+No+Contact.;Just+a+Camera+%2B+AI+on+Qualcomm+NPU;Built+for+the+Qualcomm+Snapdragon+Hackathon)
![Status](https://img.shields.io/badge/status-active%20development-brightgreen?style=for-the-badge)
![Made with](https://img.shields.io/badge/made%20with-Python%20%7C%20OpenCV%20%7C%20ONNX-blue?style=for-the-badge)
![Runtime](https://img.shields.io/badge/inference-Qualcomm%20QNN%20NPU-teal?style=for-the-badge)
![License](https://img.shields.io/badge/license-MIT-lightgrey?style=for-the-badge)
> 🩺 **No smartwatch. No sensor strap. No wires.**
> Just a phone camera, a few seconds, and an AI model that tells you what it means.
📺 ▶ Watch the Full Demo - https://drive.google.com/file/d/1vaZMRPCEzDRCv8XPaEf8EX11I0lplFBG/view?usp=sharing

---
📋 Table of Contents
The Idea
Our Pipeline
The Risk Prediction Model
Tech Stack
Devices Used
Project Structure
Getting It Running
What's Next
Team
---
💡 The Idea
Heart rate monitoring today almost always assumes you're wearing something — a smartwatch, a chest strap, a clip-on oximeter. PulseSense drops that assumption entirely.
Point a phone camera at a face, and the system:
Detects the forehead
Reads tiny, invisible color changes caused by blood flow
Turns those changes into a stable BPM reading
Feeds that BPM into a trained ML model to predict a Risk Category
Shows everything live on a dashboard
No contact. No wearable. Just vision + signal processing + machine learning.
---
🔄 Our Pipeline
```
 📱 Camera Feed
      │
      ▼
 Face Detection  ───────►  isolates the face region every frame
      │
      ▼
 Forehead ROI  ─────────►  ignores eyes/nose/mouth, keeps only forehead
      │
      ▼
 Green Channel Extraction ► blood flow shows up clearest here
      │
      ▼
 Noise Filtering  ───────►  strips out motion, blink & lighting artifacts
      │
      ▼
 Frequency Analysis  ────►  finds the repeating heartbeat rhythm
      │
      ▼
 60-Reading Smart Average ► smooths out a single BPM value
      │
      ▼
 Risk Prediction Model  ─►  MLPClassifier → Risk Category
      │
      ▼
 💻 Live Dashboard
```
---
🤖 The Risk Prediction Model
BPM alone doesn't mean much to a non-technical user, so we trained a model to turn it into an actionable Risk Category.
Dataset: Human Vital Signs Dataset
Input: Heart Rate → Output: Risk Category
Detail	Value
Framework	Scikit-learn
Algorithm	MLPClassifier
Train/Test Split	80 / 20
Evaluation Metric	Accuracy
Training pipeline:
`Dataset → Preprocessing → MLPClassifier → risk_model.pkl`
From `.pkl` to Qualcomm NPU
We didn't stop at a trained scikit-learn model — it's converted and optimized to actually run on Qualcomm hardware:
```
risk_model.pkl
     │  (export)
     ▼
risk_model.onnx
     │  (upload)
     ▼
Qualcomm AI Hub
     │  (compile + optimize)
     ▼
models/model.onnx
     │
     ▼
Runs on-device via QNN Execution Provider (NPU)
     ↳ falls back to CPU automatically if NPU runtime is unavailable
```
Inference script: `run_on_npu.py`
---
🧰 Tech Stack
Layer	Tools
Vision & Signal	Python, OpenCV, NumPy, SciPy
ML Training	Scikit-learn (MLPClassifier)
Model Deployment	ONNX, Qualcomm AI Hub, QNN Execution Provider
Dashboard	Real-time web-based visualization
---
📱 Devices Used
Device	What It Does
OnePlus 15	Captures the live camera feed, runs face detection, forehead ROI extraction, BPM calculation, and risk prediction
ThinkPad Laptop	Hosts and renders the real-time dashboard
---
📂 Project Structure
```
├── train.py               # Train the MLP risk-prediction model
├── convert_to_onnx.py     # Convert trained model to ONNX
├── inference.py           # Run inference using the trained model
├── run_on_npu.py          # Run the optimized ONNX model on Qualcomm NPU
└── models/
    ├── risk_model.pkl     # Trained Scikit-learn model
    ├── risk_model.onnx    # ONNX-converted model
    └── model.onnx         # Qualcomm AI Hub optimized model
```
---
🚀 Getting It Running
## 🚀 Quick Setup Instructions

To run this project on your local machine, follow these steps:

### 1. Clone the Repository
```bash
git clone [https://github.com/Abhigyan-Moharana/Heartbeat_Measure.git](https://github.com/Abhigyan-Moharana/Heartbeat_Measure.git)
cd Heartbeat_Measure
2. Setup Virtual Environment
Bash
# Create the environment
python -m venv .venv

# Activate the environment
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate
3. Install Dependencies
Bash
pip install -r requirements.txt
4. Configuration (Required)
Since config.py is excluded for security, you must create it manually:

Create a file named config.py in the root directory.

Add your Google Gemini API Key inside it:

Python
API_KEY = "YOUR_GEMINI_API_KEY_HERE"
5. Camera Setup
Open app.py and locate the cv2.VideoCapture line.

Update the URL with your camera's IP stream (e.g., http://192.168.x.x:8080/video).

6. Run the Application
Bash
python app.py
Open your browser and navigate to: http://127.0.0.1:5000
```
---
🔮 What's Next
Deeper health insights beyond BPM alone — HRV, stress indicators
Historical trend tracking and export
Further NPU latency optimization
Remote caregiver alerts on high-risk readings
---
👥 Team
Name	Email
Ayush Gupta	guptaaayush5678@gmail.com
Uday Ruhil	udayruhil.82@gmail.com
Abhigyan Moharana	abhigyanmohcr007@gmail.com
---
Built for the Qualcomm Snapdragon Hackathon.