import numpy as np
import time
import matplotlib.pyplot as plt
from skimage import io, color
from numpy.lib.stride_tricks import sliding_window_view

def load_image(path):
    img = io.imread(path)
    if img.ndim == 3:
        img = color.rgb2gray(img)
    return img.astype(np.float64)

def convolve2d(img, kernel):
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(img, ((ph, ph), (pw, pw)), mode='edge')
    windows = sliding_window_view(padded, (kh, kw))
    return np.sum(windows * kernel, axis=(-2, -1))

def gradient_magnitude(gx, gy):
    mag = np.sqrt(gx**2 + gy**2)
    return np.clip(mag / mag.max(), 0, 1)

def normalize(arr):
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-8)

def prewitt(img):
    Kx = np.array([[-1, 0, 1],
                   [-1, 0, 1],
                   [-1, 0, 1]], dtype=np.float64)

    Ky = np.array([[-1, -1, -1],
                   [ 0,  0,  0],
                   [ 1,  1,  1]], dtype=np.float64)

    gx = convolve2d(img, Kx)
    gy = convolve2d(img, Ky)
    return gradient_magnitude(gx, gy)

def sobel(img):
    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float64)

    Ky = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=np.float64)

    gx = convolve2d(img, Kx)
    gy = convolve2d(img, Ky)
    return gradient_magnitude(gx, gy)

def roberts(img):
    Kx = np.array([[1,  0],
                   [0, -1]], dtype=np.float64)

    Ky = np.array([[ 0, 1],
                   [-1, 0]], dtype=np.float64)

    gx = convolve2d(img, Kx)
    gy = convolve2d(img, Ky)
    return gradient_magnitude(gx, gy)

def extended_sobel(img):
    Kx = np.array([[-1, -2,  0,  2,  1],
                   [-4, -8,  0,  8,  4],
                   [-6,-12,  0, 12,  6],
                   [-4, -8,  0,  8,  4],
                   [-1, -2,  0,  2,  1]], dtype=np.float64)

    Ky = Kx.T

    gx = convolve2d(img, Kx)
    gy = convolve2d(img, Ky)
    return gradient_magnitude(gx, gy)

def kirsch(img):
    kernels = [
        np.array([[ 5,  5,  5],
                  [-3,  0, -3],
                  [-3, -3, -3]], dtype=np.float64),
        np.array([[-3,  5,  5],
                  [-3,  0,  5],
                  [-3, -3, -3]], dtype=np.float64),
        np.array([[-3, -3,  5],
                  [-3,  0,  5],
                  [-3, -3,  5]], dtype=np.float64),
        np.array([[-3, -3, -3],
                  [-3,  0,  5],
                  [-3,  5,  5]], dtype=np.float64),
        np.array([[-3, -3, -3],
                  [-3,  0, -3],
                  [ 5,  5,  5]], dtype=np.float64),
        np.array([[-3, -3, -3],
                  [ 5,  0, -3],
                  [ 5,  5, -3]], dtype=np.float64),
        np.array([[ 5, -3, -3],
                  [ 5,  0, -3],
                  [ 5, -3, -3]], dtype=np.float64),
        np.array([[ 5,  5, -3],
                  [ 5,  0, -3],
                  [-3, -3, -3]], dtype=np.float64),
    ]

    responses = np.stack([np.abs(convolve2d(img, K)) for K in kernels], axis=0)
    mag = np.max(responses, axis=0)
    return normalize(mag)

def run_all(image_path, n_runs=5):
    img = load_image(image_path)
    print(f"Image shape : {img.shape}")
    print(f"Runs per method: {n_runs}\n")

    methods = {
        "Prewitt"       : prewitt,
        "Sobel"         : sobel,
        "Roberts"       : roberts,
        "Extended Sobel": extended_sobel,
        "Kirsch"        : kirsch,
    }

    results  = {}
    runtimes = {}

    for name, fn in methods.items():
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            out = fn(img)
            times.append(time.perf_counter() - t0)
        results[name]  = out
        runtimes[name] = np.mean(times)
        print(f"{name:<18} avg runtime: {runtimes[name]*1000:.2f} ms")

    # Plot hasil
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    axes[0].imshow(img, cmap='gray')
    axes[0].set_title("Original", fontsize=13)
    axes[0].axis('off')

    for i, (name, res) in enumerate(results.items(), start=1):
        axes[i].imshow(res, cmap='gray')
        axes[i].set_title(f"{name}\n{runtimes[name]*1000:.2f} ms", fontsize=12)
        axes[i].axis('off')

    plt.suptitle("Edge Detection Comparison", fontsize=15, fontweight='bold')
    plt.tight_layout()
    # plt.savefig("edge_comparison.png", dpi=150)
    plt.show()

    # Bar chart runtime
    fig2, ax = plt.subplots(figsize=(8, 4))
    names = list(runtimes.keys())
    vals  = [runtimes[n]*1000 for n in names]
    bars  = ax.bar(names, vals, color=['#4C72B0','#DD8452','#55A868','#C44E52','#8172B2'])
    ax.set_ylabel("Avg Runtime (ms)")
    ax.set_title("Runtime Comparison")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{v:.2f}ms", ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig("runtime_comparison.png", dpi=150)
    plt.show()

    return results, runtimes