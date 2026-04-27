import cv2
import numpy as np
import matplotlib.pyplot as plt

BUFFER_SIZE  = 300   # (≈10 detik di 30fps)
PLOT_W, PLOT_H = 640, 300
PANEL_H = 200   
GAP     = 1     

BG_COLOR     = (28, 28, 28)
GRID_COLOR   = (52, 52, 52)
DIVIDER_COLOR= (70, 70, 70)


# ── Util ─────────────────────────────────────────────────────────────
def _normalize(arr, margin=10, height=PANEL_H):
    """Normalize array ke koordinat pixel (atas=kecil, bawah=besar)"""
    a_min, a_max = arr.min(), arr.max()
    if a_max - a_min < 1e-8:
        return np.full(len(arr), height // 2, dtype=int)
    norm = (arr - a_min) / (a_max - a_min)
    return (margin + (1 - norm) * (height - 2 * margin)).astype(int)


def _draw_grid(canvas, n_h=4):
    h, w = canvas.shape[:2]
    for i in range(1, n_h):
        y = int(h * i / n_h)
        cv2.line(canvas, (0, y), (w, y), GRID_COLOR, 1)


def _draw_line_signal(canvas, ys, color, thickness=1):
    w = canvas.shape[1]
    n = len(ys)
    for i in range(1, n):
        x1 = int((i-1) / (n-1) * (w-1))
        x2 = int(i     / (n-1) * (w-1))
        cv2.line(canvas, (x1, ys[i-1]), (x2, ys[i]),
                 color, thickness, cv2.LINE_AA)


def _label(canvas, text, pos, color=(180,180,180), scale=0.42, bold=False):
    cv2.putText(canvas, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 2 if bold else 1, cv2.LINE_AA)


# ── Panel 1: Raw Green Signal ─────────────────────────────────────────
def panel_raw(signal_g, width=PLOT_W, height=PANEL_H):
    canvas = np.full((height, width, 3), BG_COLOR, dtype=np.uint8)
    _draw_grid(canvas)

    _label(canvas, "RAW  (Green Channel — detrended)", (8, 18),
           color=(100, 220, 100), scale=0.45, bold=True)

    if len(signal_g) < 2:
        _label(canvas, "Collecting...", (20, height//2), color=(120,120,120))
        return canvas

    arr = np.array(signal_g, dtype=np.float32)
    ys  = _normalize(arr, height=height)
    _draw_line_signal(canvas, ys, color=(80, 210, 80), thickness=1)

    # Anotasi min/max
    _label(canvas, f"max:{arr.max():.2f}", (width-100, 18), color=(140,140,140))
    _label(canvas, f"min:{arr.min():.2f}", (width-100, height-6), color=(140,140,140))
    _label(canvas, f"n={len(arr)}", (8, height-6), color=(100,100,100))
    return canvas


# ── Panel 2: Bandpass Signal (morfologi PPG) ──────────────────────────
def panel_bandpass(sig_bp, bpm, width=PLOT_W, height=PANEL_H):
    canvas = np.full((height, width, 3), BG_COLOR, dtype=np.uint8)
    _draw_grid(canvas)

    if sig_bp is None or len(sig_bp) < 2:
        _label(canvas, "Processing...", (20, height//2), color=(120,120,120))
        return canvas

    ys = _normalize(sig_bp, height=height)
    _draw_line_signal(canvas, ys, color=(80, 160, 255), thickness=1)

    # Gambar zero-line
    zero_y = _normalize(np.array([0.0, 0.0]), height=height)
    mid_y  = int((height * 0.1 + height * 0.9) / 2)  # approx center
    cv2.line(canvas, (0, mid_y), (width, mid_y), (60, 60, 80), 1)

    # BPM label
    if bpm is not None:
        color  = (80, 255, 80) if 50 <= bpm <= 150 else (80, 80, 255)
        _label(canvas, f"HR  {bpm:.1f} BPM", (width//2, 20),
               color=color, scale=0.6, bold=True)

    # Amplitudo (proxy kekuatan sinyal)
    amp = sig_bp.max() - sig_bp.min()
    _label(canvas, f"amp:{amp:.4f}", (width-100, 18), color=(140,140,140))
    return canvas


# ── Panel 3: FFT Spectrum ─────────────────────────────────────────────
def panel_fft(freqs, power, bpm, width=PLOT_W, height=PANEL_H):
    canvas = np.full((height, width, 3), BG_COLOR, dtype=np.uint8)
    _draw_grid(canvas)

    if freqs is None or power is None:
        _label(canvas, "Collecting...", (20, height//2), color=(120,120,120))
        return canvas

    low_hz  = 0.7
    high_hz = 4
    mask    = (freqs >= low_hz) & (freqs <= high_hz)

    freqs_v = freqs[mask]
    power_v = power[mask]

    if len(power_v) < 2:
        return canvas

    p_min, p_max = power_v.min(), power_v.max()
    if p_max - p_min < 1e-10:
        return canvas
    power_norm = (power_v - p_min) / (p_max - p_min)

    # Gambar bar chart
    n     = len(power_norm)
    bar_w = max(1, width // n)
    peak_idx = np.argmax(power_norm)

    for i, val in enumerate(power_norm):
        x1    = int(i / n * width)
        x2    = min(x1 + bar_w, width - 1)
        y_top = int((1 - val) * (height - 25)) + 10
        y_bot = height - 20

        is_peak = (i == peak_idx)
        color   = (60, 220, 120) if is_peak else (60, 120, 80)
        cv2.rectangle(canvas, (x1, y_top), (x2, y_bot), color, -1)

        # Garis peak
        if is_peak:
            cv2.line(canvas, (x1, 5), (x1, y_bot), (100, 255, 140), 1)

    # Sumbu X: label frekuensi + BPM
    for hz in np.arange(0.8, 3.1, 0.2):
        if low_hz <= hz <= high_hz:
            x_pos = int((hz - low_hz) / (high_hz - low_hz) * width)
            bpm_v = hz * 60
            cv2.line(canvas, (x_pos, height-18), (x_pos, height-12), (90,90,90), 1)
            _label(canvas, f"{bpm_v:.0f}", (x_pos - 10, height - 3),
                   color=(110, 110, 110), scale=0.32)
    return canvas


# ── Gabungkan 3 Panel ─────────────────────────────────────────────────
def build_visualizer(signal_g, sig_raw, sig_bp, bpm, freqs, power,
                     width=PLOT_W, height=PANEL_H):
    divider = np.full((GAP, width, 3), DIVIDER_COLOR, dtype=np.uint8)

    p1 = panel_raw(signal_g, width, height)
    p2 = panel_bandpass(sig_bp, bpm, width, height)
    p3 = panel_fft(freqs, power, bpm, width, height)

    return np.vstack([p1, divider, p2, divider, p3])


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