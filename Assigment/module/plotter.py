import cv2
import numpy as np
import matplotlib.pyplot as plt

BUFFER_SIZE  = 300   # (≈10 detik di 30fps)
PLOT_W, PLOT_H = 640, 300

def drawSignalPlot(sig_r, sig_g, sig_b, width=PLOT_W, height=PLOT_H):
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (30, 30, 30)  

    for i in range(1, 4):
        y_grid = int(height * i / 4)
        cv2.line(canvas, (0, y_grid), (width, y_grid), (60, 60, 60), 1)

    def draw_signal(signal, color, label, label_pos):
        if len(signal) < 2:
            return
        arr = np.array(signal, dtype=np.float32)

        s_min, s_max = arr.min(), arr.max()
        if s_max - s_min < 1e-6:
            return
        arr_norm = (arr - s_min) / (s_max - s_min)

        n = len(arr_norm)
        for i in range(1, n):
            x1 = int((i - 1) / (BUFFER_SIZE - 1) * (width - 1))
            x2 = int(i       / (BUFFER_SIZE - 1) * (width - 1))
            y1 = int((1 - arr_norm[i - 1]) * (height - 20)) + 10
            y2 = int((1 - arr_norm[i])     * (height - 20)) + 10
            cv2.line(canvas, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

        cv2.putText(canvas, label, label_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    draw_signal(sig_r, (80,  80,  255), "R", (width - 25, 20))
    draw_signal(sig_g, (80,  200, 80),  "G", (width - 25, 40))
    draw_signal(sig_b, (255, 120, 80),  "B", (width - 25, 60))

    cv2.putText(canvas, f"samples: {len(sig_g)}", (8, height - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    return canvas