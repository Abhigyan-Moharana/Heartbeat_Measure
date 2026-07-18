import cv2
import numpy as np
import time
from scipy.signal import butter, filtfilt, detrend

# Load the pre-trained Haar Cascade model for face detection
# Ensure this XML file is in the same directory as this script
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

# Connect to the IP Webcam
cap = cv2.VideoCapture('http://192.168.137.111:8080/video')

# Data buffers and smoothing variables
times = []
data_buffer = []
BUFFER_SIZE = 150  # Number of frames to collect before calculating the FFT
smoothed_box = None
bpm_history = []   # Tracks past readings for hyper-accuracy

t0 = time.time()

print("Starting camera... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    
    # SHIELD 1: Network Lag Protection
    # If the Wi-Fi drops a packet, skip it instead of crashing.
    if not ret or frame is None:
        continue
        
    # Check if the frame is actually valid before processing.
    if frame.shape[0] == 0 or frame.shape[1] == 0:
        continue
        
    # Convert frame to grayscale for the face detection algorithm
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # --- SMOOTHING LOGIC FOR THE BOUNDING BOX ---
        if smoothed_box is None:
            smoothed_box = [x, y, w, h]
        else:
            # 20% new position, 80% old position to eliminate micro-jitters
            smoothed_box = [int(0.2 * curr + 0.8 * prev) for curr, prev in zip([x, y, w, h], smoothed_box)]
            
        sx, sy, sw, sh = smoothed_box
        
        # Draw a blue bounding box around the smoothed face
        cv2.rectangle(frame, (sx, sy), (sx+sw, sy+sh), (255, 0, 0), 2)
        
        # SHIELD 2: Boundary Clamping
        # Prevents the script from crashing if your forehead goes off the edge of the screen
        fh_x = max(0, sx + int(sw * 0.25))
        fh_y = max(0, sy + int(sh * 0.1))
        fh_w = min(frame.shape[1] - fh_x, int(sw * 0.5))
        fh_h = min(frame.shape[0] - fh_y, int(sh * 0.15))
        
        # If the box collapses entirely, skip this frame
        if fh_w <= 0 or fh_h <= 0:
            continue
        
        # Draw a green bounding box around the forehead ROI
        cv2.rectangle(frame, (fh_x, fh_y), (fh_x+fh_w, fh_y+fh_h), (0, 255, 0), 2)
        
        # Extract the forehead region from the frame
        roi = frame[fh_y:fh_y+fh_h, fh_x:fh_x+fh_w]
        
        # Extract both Green (index 1) and Red (index 2) channels
        green_channel = roi[:, :, 1]
        red_channel = roi[:, :, 2]
        
        # Calculate the means
        green_mean = np.mean(green_channel)
        red_mean = np.mean(red_channel)
        
        # Subtract Red from Green to cancel out environmental lighting
        balanced_signal = green_mean - (0.5 * red_mean)
        
        # Record the time and the enhanced signal value
        times.append(time.time() - t0)
        data_buffer.append(balanced_signal)
        
        # Maintain a rolling window of data
        if len(data_buffer) > BUFFER_SIZE:
            data_buffer.pop(0)
            times.pop(0)
            
            # SHIELD 3: Math "God Mode"
            # If the filters hit a math anomaly, it skips the frame instead of crashing the app
            try:
                # --- SIGNAL PROCESSING (RED-GREEN + DC NORMALIZATION) ---
                signal = np.array(data_buffer)
                
                # Prevent ZeroDivisionError if network drops multiple frames in the exact same millisecond
                time_diff = times[-1] - times[0]
                if time_diff == 0:
                    continue
                    
                fps = len(times) / time_diff
                
                # 1. DC Normalization (Removes absolute lighting intensity)
                signal = (signal / np.mean(signal)) - 1.0
                
                # 2. Detrend to remove slow drifts from posture shifts
                signal = detrend(signal)
                
                # 3. Butterworth Bandpass Filter (isolates 48 to 180 BPM)
                nyquist = 0.5 * fps
                low = 0.8 / nyquist
                high = 3.0 / nyquist
                
                if 0 < low < high < 1:
                    b, a = butter(2, [low, high], btype='band')
                    signal = filtfilt(b, a, signal)
                
                # 4. Apply a Hamming Window to reduce spectral leakage
                window = np.hamming(len(signal))
                windowed_signal = signal * window
                
                # 5. Compute the Fast Fourier Transform (FFT) with Zero-Padding
                padded_length = 1024  
                
                fft_data = np.abs(np.fft.rfft(windowed_signal, n=padded_length))
                fft_freqs = np.fft.rfftfreq(padded_length, d=1.0/fps)
                
                # 6. Filter for standard human heart rates (0.8 Hz to 3.0 Hz -> 48 BPM to 180 BPM)
                valid_indices = np.where((fft_freqs > 0.8) & (fft_freqs < 3.0))
                
                if len(valid_indices[0]) > 0:
                    valid_fft = fft_data[valid_indices]
                    valid_freqs = fft_freqs[valid_indices]
                    
                    # 7. Extract the dominant frequency
                    dominant_freq = valid_freqs[np.argmax(valid_fft)]
                    raw_bpm = dominant_freq * 60
                    
                    # --- TEMPORAL SMOOTHING FOR THE DASHBOARD ---
                    # This stops the numbers from violently flickering
                    bpm_history.append(raw_bpm)
                    if len(bpm_history) > 30:  # Average the last 30 readings
                        bpm_history.pop(0)
                    
                    final_bpm = np.mean(bpm_history)
                    
                    # Display the smoothed BPM on the video feed
                    cv2.putText(frame, f"BPM: {int(final_bpm)}", (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            except Exception as e:
                # Silently pass over any math errors
                pass
                
    # Show the live feed
    cv2.imshow('rPPG Webcam Test', frame)
    
    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up resources
cap.release()
cv2.destroyAllWindows()