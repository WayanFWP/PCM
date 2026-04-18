import numpy as np
import cv2
import time
from module.segmentation import *

cap = cv2.VideoCapture(1)
face_cascade = "dataset\lecture\haarcascade_frontalface_default.xml"
eye_cascade = "dataset\lecture\haarcascade_eye.xml"
prev_time = 0

while True:
    _, frame = cap.read()
    
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    bbox = faceBBOX(frame, face_cascade, eye_cascade)
    
    if bbox != None:
        break
        
    cv2.imshow("0", bbox)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
cv2.release()
cv2.destroyAllWindows()