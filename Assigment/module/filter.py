import numpy as np
from scipy.signal import butter, filtfilt, detrend

HR_MIN_BPM = 40
HR_MAX_BPM = 180

def bandpass_filter(signal, fps, low_bpm=HR_MIN_BPM, high_bpm=HR_MAX_BPM, order=4):
    nyq     = fps / 2.0
    low_hz  = (low_bpm  / 60.0) / nyq
    high_hz = (high_bpm / 60.0) / nyq
    low_hz  = np.clip(low_hz,  1e-4, 0.99)
    high_hz = np.clip(high_hz, 1e-4, 0.99)
    if low_hz >= high_hz:
        return signal
    b, a = butter(order, [low_hz, high_hz], btype='band')
    return filtfilt(b, a, signal)


def process_signal(signal_g, timestamps):
    """
    Return semua tahapan sinyal sekaligus.

    Returns:
        sig_raw      : green channel setelah detrend
        sig_bandpass : setelah bandpass filter
        bpm          : estimasi heart rate
        freqs        : array frekuensi FFT (Hz)
        power        : power spectrum FFT
    """
    sig = np.array(signal_g,  dtype=np.float32)
    ts  = np.array(timestamps, dtype=np.float32)

    if len(sig) < 30:
        return None, None, None, None, None

    fps = len(ts) / (ts[-1] - ts[0] + 1e-9)

    # ── 1. Detrend ───────────────────────────────────────────────────
    sig_raw = detrend(sig)

    # ── 2. Bandpass ──────────────────────────────────────────────────
    sig_bp  = bandpass_filter(sig_raw, fps)

    # ── 3. FFT dari sinyal bandpass ──────────────────────────────────
    N        = len(sig_bp)
    window   = np.hanning(N)
    fft_vals = np.fft.rfft(sig_bp * window, n=N)
    freqs    = np.fft.rfftfreq(N, d=1.0/fps)
    power    = np.abs(fft_vals) ** 2

    # ── 4. Peak di rentang HR ────────────────────────────────────────
    low_hz  = HR_MIN_BPM / 60.0
    high_hz = HR_MAX_BPM / 60.0
    mask    = (freqs >= low_hz) & (freqs <= high_hz)

    bpm = None
    if np.any(mask):
        peak_freq = freqs[mask][np.argmax(power[mask])]
        bpm       = peak_freq * 60.0

    return sig_raw, sig_bp, bpm, freqs, power