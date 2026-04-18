import numpy as np
import cv2

def faceBBOX(img, path1, path2=None):
    face = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(path1)
    face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)

    eye_cascade = None
    if path2 is not None:
        eye_cascade = cv2.CascadeClassifier(path2)

    raw_bboxes = []  

    for (x, y, w, h) in face_rects:
        cv2.rectangle(face, (x, y), (x+w, y+h), (255, 255, 255), 2)

        if eye_cascade is not None:
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
            eye_data = []

            for (ex, ey, ew, eh) in eyes:
                center = (x + ex + ew//2, y + ey + eh//2)
                radius = ew // 2
                cv2.circle(face, center, radius, (255, 255, 255), 2)
                eye_data.append((center, radius))

            if len(eye_data) == 2:
                (c1, r1), (c2, r2) = eye_data

                left_x      = min(c1[0], c2[0])
                right_x     = max(c1[0], c2[0])
                forehead_top = y

                higher_eye    = c1 if c1[1] < c2[1] else c2
                higher_radius = r1 if c1[1] < c2[1] else r2
                forehead_bottom = higher_eye[1] - higher_radius

                if forehead_top < forehead_bottom:
                    fw = right_x - left_x
                    fh = forehead_bottom - forehead_top
                    raw_bboxes.append((left_x, forehead_top, fw, fh))

    return face, raw_bboxes


def extractSignal(roi):
    if roi is None or roi.size == 0:
        return None
    
    mean_rgb = cv2.mean(roi)[:3]  
    r, g, b = mean_rgb[2], mean_rgb[1], mean_rgb[0] 
    return r, g, b