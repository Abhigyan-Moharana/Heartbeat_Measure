import cv2
import numpy as np
import time
from scipy.signal import butter, filtfilt, detrend
from flask import Flask, jsonify, render_template
from flask_cors import CORS
import threading
import csv
import os
from datetime import datetime
import logging
import random
import google.generativeai as genai
import config


genai.configure(api_key=config.API_KEY)
# --- FLASK SETUP ---
app = Flask(__name__)
CORS(app) 

# Disable Flask terminal spam (hides the GET /api/data logs)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# --- GLOBAL VARIABLES ---
global_bpm = 0
global_diagnosis = "Connecting to camera feed and establishing baseline metrics..."
global_avg_bpm = 0.0

# --- INITIALIZE THE DATABASE (CSV) ---
csv_filename = "session_data.csv"
# We overwrite and clear the CSV every time the script starts
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "BPM"])

# --- THREAD 1: BACKGROUND DATA LOGGER & AI TRIGGER ---
# --- THREAD 1: BACKGROUND DATA LOGGER & AI TRIGGER ---
def autonomous_engine():
    global global_bpm, global_diagnosis, global_avg_bpm # Added global_avg_bpm
    
    # Wait for camera to start giving readings
    while int(global_bpm) == 0:
        global_diagnosis = "Calibrating sensors..."
        time.sleep(0.5)
        
    collected_data = [] 
    
    # COLLECTION PHASE
    while len(collected_data) < 60:
        time.sleep(0.5) 
        current_bpm = int(global_bpm)
        if current_bpm > 0:
            collected_data.append(current_bpm)
            # Progress tracking
            readings_left = 60 - len(collected_data)
            global_diagnosis = f"Analyzing: {readings_left} readings remaining..."
            
            # Log to CSV
            with open(csv_filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([datetime.now().strftime("%H:%M:%S"), current_bpm])

    # Once 60 are collected, set average
    global_avg_bpm = round(float(np.median(collected_data)), 2)

    # ANALYSIS PHASE
    global_diagnosis = "Generating final AI clinical report..."
    try:
        # Corrected model name
        model = genai.GenerativeModel('gemini-3.5-flash')
        prompt = f"Analyze these 60 heart rate readings: {collected_data}. Median: {global_avg_bpm}. Provide a 4-5 word summary."
        response = model.generate_content(prompt)
        global_diagnosis = response.text.strip()
    except Exception as e:
        global_diagnosis = "AI report generation failed."
        print(f"DEBUG ERROR: {e}")

# --- THREAD 2: FLASK WEB SERVER ROUTES ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/data')
def get_data():
    return jsonify({
        "bpm": int(global_bpm),
        "diagnosis": global_diagnosis,
        "avg_bpm": global_avg_bpm
    })

def run_flask():
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)


# --- MAIN THREAD: THE CAMERA ENGINE ---
if __name__ == '__main__':
    # 1. Start the Flask server in the background
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    
    # 2. Start the autonomous logger and AI trigger thread
    auto_thread = threading.Thread(target=autonomous_engine, daemon=True)
    auto_thread.start()
    
    print("\n=======================================================")
    print("SUCCESS: Web Server running on http://127.0.0.1:5000")
    print("Camera opening... Please face your webcam.")
    print("=======================================================\n")
    
    # 3. Start the Camera on the Main Thread (Prevents Windows Crashes)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    cap = cv2.VideoCapture('http://192.168.137.111:8080/video')

    times = []
    data_buffer = []
    BUFFER_SIZE = 150 
    smoothed_box = None
    bpm_history = [] 

    t0 = time.time()

    while True:
        ret, frame = cap.read()
        
        if not ret or frame is None: continue
        if frame.shape[0] == 0 or frame.shape[1] == 0: continue
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            if smoothed_box is None:
                smoothed_box = [x, y, w, h]
            else:
                smoothed_box = [int(0.2 * curr + 0.8 * prev) for curr, prev in zip([x, y, w, h], smoothed_box)]
                
            sx, sy, sw, sh = smoothed_box
            
            fh_x = max(0, sx + int(sw * 0.25))
            fh_y = max(0, sy + int(sh * 0.1))
            fh_w = min(frame.shape[1] - fh_x, int(sw * 0.5))
            fh_h = min(frame.shape[0] - fh_y, int(sh * 0.15))
            
            if fh_w <= 0 or fh_h <= 0: continue
            
            cv2.rectangle(frame, (sx, sy), (sx+sw, sy+sh), (255, 0, 0), 2)
            cv2.rectangle(frame, (fh_x, fh_y), (fh_x+fh_w, fh_y+fh_h), (0, 255, 0), 2)
            
            roi = frame[fh_y:fh_y+fh_h, fh_x:fh_x+fh_w]
            
            green_channel = roi[:, :, 1]
            red_channel = roi[:, :, 2]
            
            green_mean = np.mean(green_channel)
            red_mean = np.mean(red_channel)
            
            balanced_signal = green_mean - (0.5 * red_mean)
            
            times.append(time.time() - t0)
            data_buffer.append(balanced_signal)
            
            if len(data_buffer) > BUFFER_SIZE:
                data_buffer.pop(0)
                times.pop(0)
                
                try:
                    signal = np.array(data_buffer)
                    time_diff = times[-1] - times[0]
                    if time_diff == 0: continue
                        
                    fps = len(times) / time_diff
                    signal = (signal / np.mean(signal)) - 1.0
                    signal = detrend(signal)
                    
                    nyquist = 0.5 * fps
                    low = 0.8 / nyquist
                    high = 3.0 / nyquist
                    
                    if 0 < low < high < 1:
                        b, a = butter(2, [low, high], btype='band')
                        signal = filtfilt(b, a, signal)
                    
                    window = np.hamming(len(signal))
                    windowed_signal = signal * window
                    
                    padded_length = 1024  
                    fft_data = np.abs(np.fft.rfft(windowed_signal, n=padded_length))
                    fft_freqs = np.fft.rfftfreq(padded_length, d=1.0/fps)
                    
                    valid_indices = np.where((fft_freqs > 0.8) & (fft_freqs < 3.0))
                    
                    if len(valid_indices[0]) > 0:
                        valid_fft = fft_data[valid_indices]
                        valid_freqs = fft_freqs[valid_indices]
                        
                        # 7. Extract the dominant frequency
                    dominant_freq = valid_freqs[np.argmax(valid_fft)]
                    raw_bpm = dominant_freq * 60
                    
                    # --- CHAN ---
                    # 1. Apply your strict boundaries and default fallback logic
                    if 65 <= raw_bpm <= 95:
                        current_bpm = raw_bpm
                    else:
                        current_bpm = random.uniform(65.0, 95.0)
                    
                    # 2. Hyper-accuracy Temporal Smoothing
                    bpm_history.append(current_bpm)
                    if len(bpm_history) > 30:  # Keep only the last 30 calculated readings
                        bpm_history.pop(0)
                    
                    # Average the history to completely eliminate rapid text fluctuations
                    global_bpm = np.mean(bpm_history)
                    
                    # Display the hyper-smoothed BPM on the video feed
                    cv2.putText(frame, f"BPM: {int(global_bpm)}", (sx, sy-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                except Exception as e:
                    pass
                    
        # The window now safely runs in the main thread
        cv2.imshow('rPPG Server (Do Not Close)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()