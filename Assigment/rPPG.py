import numpy as np
import cv2
import time
from collections import deque
from module.segmentation import *
from module.smoothing import KalmanFilter
from module.plotter import *
from module.filter import *

cap          = cv2.VideoCapture(1)
face_cascade = "dataset/lecture/haarcascade_frontalface_default.xml"
eye_cascade  = "dataset/lecture/haarcascade_eye.xml"
prev_time    = 0
start_time   = time.time()

signal_r   = deque(maxlen=BUFFER_SIZE)
signal_g   = deque(maxlen=BUFFER_SIZE)
signal_b   = deque(maxlen=BUFFER_SIZE)
timestamps = deque(maxlen=BUFFER_SIZE)

FFT_INTERVAL = 15
frame_count  = 0

# State hasil proses sinyal
state = dict(sig_raw=None, sig_bp=None, bpm=None, freqs=None, power=None)

forehead_kf = KalmanFilter()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    curr_time = time.time()
    fps       = 1 / (curr_time - prev_time + 1e-9)
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    bbox_frame, raw_bboxes = faceBBOX(frame, face_cascade, eye_cascade)

    smoothed = forehead_kf.update(raw_bboxes[0]) if raw_bboxes else forehead_kf.predict_only()

    display = bbox_frame.copy()

    if smoothed is not None:
        sx, sy, sw, sh = smoothed
        sx = max(0, sx);  sy = max(0, sy)
        sw = min(sw, frame.shape[1] - sx)
        sh = min(sh, frame.shape[0] - sy)

        if sw > 0 and sh > 0:
            cv2.rectangle(display, (sx, sy), (sx+sw, sy+sh), (0, 255, 0), 2)

            forehead = frame[sy:sy+sh, sx:sx+sw]
            if forehead.size > 0:
                cv2.imshow("Forehead ROI", forehead)

                result = extractSignal(forehead)
                if result is not None:
                    r, g, b = result
                    signal_r.append(r)
                    signal_g.append(g)
                    signal_b.append(b)
                    timestamps.append(time.time() - start_time)

    cv2.imshow("Camera", display)
    
    
    frame_count += 1
    if frame_count % FFT_INTERVAL == 0 and len(signal_g) > 60:
        sig_raw, sig_bp, bpm, freqs, power = process_signal(signal_g, timestamps)
        state.update(sig_raw=sig_raw, sig_bp=sig_bp,
                     bpm=bpm, freqs=freqs, power=power)
    if len(signal_g) > 1:
        vis = build_visualizer(
            signal_g          = signal_g,
            sig_raw           = state['sig_raw'],
            sig_bp            = state['sig_bp'],
            bpm               = state['bpm'],
            freqs             = state['freqs'],
            power             = state['power'],
        )
        cv2.imshow("rPPG Analyzer", vis)

    # if len(signal_g) > 1:
    #     signal_plot = drawSignalPlot(signal_r, signal_g, signal_b)
                
    #     cv2.imshow("rPPG Signal", signal_plot)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()