import numpy as np
import cv2
import matplotlib.pyplot as plt

def faceBBOX(img, path1, path2=None):
    face = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(path1)
    face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)

    eye_cascade = None
    if path2 is not None:
        eye_cascade = cv2.CascadeClassifier(path2)
    
    acc_roi = []

    for (x,y,w,h) in face_rects:
        cv2.rectangle(face, (x,y), (x+w,y+h), (255,255,255), 2)

        if eye_cascade is not None:
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)

            for (ex,ey,ew,eh) in eyes:
                center = (x + ex + ew//2, y + ey + eh//2)
                radius = ew // 2
                cv2.circle(face, center, radius, (255,255,255), 2)

    return face

# def 