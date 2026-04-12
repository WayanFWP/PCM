import imageio
import matplotlib.pyplot as plt
import numpy as np

def CNR(inside, outside):
    mean_in  = np.mean(inside)
    std_in   = np.std(inside)
    mean_out = np.mean(outside)
    std_out  = np.std(outside)
    
    cnr = abs(mean_in - mean_out) / np.sqrt(0.5 * (std_in**2 + std_out**2)) \
        if (std_in > 0 and std_out > 0) else 0    
    print(f"CNR result : {cnr}")
    
    return cnr

def ENL(inside, outside):
    mean_in  = np.mean(inside)
    std_in   = np.std(inside)
    mean_out = np.mean(outside)
    std_out  = np.std(outside)
    
    enl = (mean_in / std_in) ** 2 if std_in > 0 else 0
    
    print(f"ENL Result: {enl}")
    return enl

def PSNR(label, original, processed):
    mse = np.mean((original - processed) ** 2)
    psnr = 10 * np.log10((255**2) / mse)

    print(f"{label} PSNR:", psnr)
    
    return psnr
    
def MSE(label, original, processed):
    original = original.astype(np.float64)
    processed = processed.astype(np.float64)

    mse = np.mean((original - processed) ** 2)

    print("MSE:", mse)
    
    return mse
    
def AnalysisPreROI(label, original, proccessed):
    PSNR(label, original, proccessed)
    MSE(label, original, proccessed)
    
