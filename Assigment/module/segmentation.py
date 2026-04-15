import numpy as np
import cv2
import matplotlib.pyplot as plt

def faceBBOX(img, path):
    face = img.copy()
    face_cascade = cv2.CascadeClassifier(path)
    face_rects = face_cascade.detectMultiscale(face)
    
    for(x,y,w,h) in face_rects:
        cv2.rectangle(face, (x,y), (x+w,y+h), (255,255,255), 10) 
        
    return face

