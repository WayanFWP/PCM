import cv2
import numpy as np

class KalmanFilter:
    def __init__(self):
        # State: [x, y, w, h, vx, vy, vw, vh]
        # 4 position + 4 velocity
        self.kf = cv2.KalmanFilter(8, 4)

        # Transition matrix (constant movement)
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ], dtype=np.float32)

        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ], dtype=np.float32)

        # Process noise (Q) → makin besar = lebih responsif tapi kurang smooth
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 0.03

        # Measurement noise (R) → makin besar = lebih smooth tapi lebih lag
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1.0

        # Initial error covariance 
        self.kf.errorCovPost = np.eye(8, dtype=np.float32)

        self.initialized = False
        self.last_prediction = None

    def init_state(self, bbox):
        x, y, w, h = bbox
        self.kf.statePost = np.array(
            [[x], [y], [w], [h], [0], [0], [0], [0]], 
            dtype=np.float32
        )
        self.initialized = True

    def update(self, bbox):
        if not self.initialized:
            self.init_state(bbox)
            return bbox

        x, y, w, h = bbox
        measurement = np.array([[x], [y], [w], [h]], dtype=np.float32)

        self.kf.predict()
        corrected = self.kf.correct(measurement)

        result = corrected[:4].flatten()
        self.last_prediction = result
        return result.astype(int)

    def predict_only(self):
        if not self.initialized:
            return None

        predicted = self.kf.predict()
        result = predicted[:4].flatten()
        self.last_prediction = result
        return result.astype(int)