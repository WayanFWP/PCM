import numpy as np
from scipy.signal import find_peaks

def LPF(signal, fl, fs):
    N = len(signal)
    T = 1 / fs
    Wc = 2 * np.pi * fl

    denom = (4 / T**2) + (2 * np.sqrt(2) * Wc / T) + Wc**2
    b1 = ((8 / T**2) - (2 * Wc**2)) / denom
    b2 = ((4 / T**2) - (2 * np.sqrt(2) * Wc / T) + Wc**2) / denom
    a0 = Wc**2 / denom
    a1 = 2 * Wc**2 / denom
    a2 = a0
    y = np.zeros(N)
    for n in range(2, N-2):
        y[n] = (b1 * y[n-1]) - (b2 * y[n-2]) + (a0 * signal[n]) + (a1 * signal[n-1]) + (a2 * signal[n-2])
    return y
  
def HPF(signal,fh,fs):
    N = len(signal)
    T = 1/fs
    Wc = 2 * np.pi * fh

    denom = (4/T**2) + (2*np.sqrt(2)*Wc/T) + Wc**2
    b1 = ((8/T**2) - 2*Wc**2)/ denom
    b2 = ((4/T**2) - (2*np.sqrt(2)*Wc/T) + Wc**2)/ denom
    a0 = (4/T**2) / denom
    a1 = (-8/T**2) / denom
    a2 = a0
    y = np.zeros(N)
    for n in range(2, N-1):
        y[n] = (b1 * y[n-1]) - (b2 * y[n-2]) + (a0 * signal[n]) + (a1 * signal[n-1]) + (a2 * signal[n-2])
    return y
    
# cutoff =  0.65–4 Hz.
def BPF(signal, fc1, fc2, fs):
    lpf_data = LPF(signal, fc1, fs)
    filtered_data = HPF(lpf_data, fc2, fs)
    return filtered_data

def bandpass(signal, lowcut, highcut, fs):
    from scipy.signal import butter, filtfilt
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(6, [low, high], btype='band')
    return filtfilt(b, a, signal)

def fft(signal, fs):
    N = len(signal)
    window = np.hanning(N)
    fft_vals = np.fft.rfft(signal * window, n=N)
    freqs = np.fft.rfftfreq(N, d=1.0/fs)
    power = np.abs(fft_vals) ** 2
    return freqs, power

def runRGB2BPM(signal_g, timestamps):
    sig = np.array(signal_g, dtype=np.float32)
    ts = np.array(timestamps, dtype=np.float32)
    
    fps = len(ts) / (ts[-1] - ts[0] + 1e-9)
    
    norm = (sig - np.mean(sig)) / np.mean(sig)
    
    # filtering = BPF(norm, 0.65, 4.0, fps)
    filtering = bandpass(norm, 0.65, 4.0, fps)
    
    freqs, power = fft(filtering, fps)
    
    peaks, _ = find_peaks(filtering, distance=fps*0.5, height=np.mean(filtering))
    rr_intervals = np.diff(peaks) / fps
    bpm_interval = 60/rr_intervals
    
    bpm = np.mean(bpm_interval) if len(bpm_interval) > 0 else None
    
    return norm, filtering, bpm, freqs, power