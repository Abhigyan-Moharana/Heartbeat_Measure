import cv2
import numpy as np
import time

# Load the pre-trained Haar Cascade model for face detection
# Ensure this XML file is in the same directory as this script
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

# Initialize the laptop's default webcam (Index 0)
cap = cv2.VideoCapture(0)

# Data buffers for time and signal values
times = []
data_buffer = []
BUFFER_SIZE = 150  # Number of frames to collect before calculating the FFT

t0 = time.time()

print("Starting camera... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
        
    # Convert frame to grayscale for the face detection algorithm
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw a blue bounding box around the detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Define the Region of Interest (ROI) for the forehead
        # We target the upper middle section of the bounding box
        fh_x = x + int(w * 0.25)
        fh_y = y + int(h * 0.1)
        fh_w = int(w * 0.5)
        fh_h = int(h * 0.15)
        
        # Draw a green bounding box around the forehead ROI
        cv2.rectangle(frame, (fh_x, fh_y), (fh_x+fh_w, fh_y+fh_h), (0, 255, 0), 2)
        
        # Extract the forehead region from the frame
        roi = frame[fh_y:fh_y+fh_h, fh_x:fh_x+fh_w]
        
        # Isolate the Green Channel (OpenCV loads images in BGR format, so index 1 is Green)
        # Blood absorbs green light, making it the strongest channel for rPPG
        green_channel = roi[:, :, 1]
        green_mean = np.mean(green_channel)
        
        # Record the time and the mean color value
        times.append(time.time() - t0)
        data_buffer.append(green_mean)
        
        # Maintain a rolling window of data
        if len(data_buffer) > BUFFER_SIZE:
            data_buffer.pop(0)
            times.pop(0)
            
            # --- Signal Processing ---
            # 1. Normalize the signal to zero mean and unit variance
            signal = np.array(data_buffer)
            signal = (signal - np.mean(signal)) / np.std(signal)
            
            # 2. Compute the Fast Fourier Transform (FFT)
            window = np.hamming(len(signal))
            windowed_signal = signal * window

            fps = len(times) / (times[-1] - times[0])
            padded_length = 1024  
            
            fft_data = np.abs(np.fft.rfft(windowed_signal, n=padded_length))
            fft_freqs = np.fft.rfftfreq(padded_length, d=1.0/fps)
            
            # 3. Filter for standard human heart rates (0.8 Hz to 3.0 Hz -> 48 BPM to 180 BPM)
            valid_indices = np.where((fft_freqs > 0.8) & (fft_freqs < 3.0))
            
            if len(valid_indices[0]) > 0:
                valid_fft = fft_data[valid_indices]
                valid_freqs = fft_freqs[valid_indices]
                
                # 4. Extract the dominant frequency
                dominant_freq = valid_freqs[np.argmax(valid_fft)]
                
                # 5. Convert frequency (Hz) to Beats Per Minute (BPM)
                bpm = dominant_freq * 60
                
                # Display the BPM on the video feed
                cv2.putText(frame, f"BPM: {int(bpm)}", (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
    # Show the live feed
    cv2.imshow('rPPG Webcam Test', frame)
    
    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up resources
cap.release()
cv2.destroyAllWindows()