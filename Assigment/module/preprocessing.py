import imageio
import matplotlib.pyplot as plt
import numpy as np

def CNR(label, signals, bg):
    S = np.mean(signals)
    B = np.mean(bg)
    
    sigma_S = np.std(signals)
    sigma_B = np.std(bg)
    
    cnr = abs(S-B)/np.sqrt(sigma_S**2+sigma_B**2)
    
    print(f"{label} CNR result : {cnr}")

def ENL(im, label):
    mu = np.mean(im)
    sigma = np.std(im)
    
    enl = (mu**2) / (sigma**2)
    
    print(f"{label} ENL Result: {enl}")

def PSNR(label, original, processed):
    mse = np.mean((original - processed) ** 2)
    psnr = 10 * np.log10((255**2) / mse)

    print(f"{label} PSNR:", psnr)
    
def MSE(label, original, processed):
    original = original.astype(np.float64)
    processed = processed.astype(np.float64)

    mse = np.mean((original - processed) ** 2)

    print("MSE:", mse)
    
def AnalysisPreROI(label, original, proccessed):
    PSNR(label, original, proccessed)
    MSE(label, original, proccessed)