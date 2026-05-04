import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi

from skimage import exposure

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
    
def NM(label, original, processed):
    M_Orig = np.mean(original)
    M_Pro = np.mean(processed)
    
    nm = M_Pro / M_Orig
    
    print(f"{label}-NM : {nm}")
    
    return nm
    
def AnalysisPreROI(label, original, proccessed):
    psnr = PSNR(label, original, proccessed)
    mse = MSE(label, original, proccessed)
    nm = NM(label, original, proccessed)
    
    return psnr, mse, nm

    
class Analysis:
    def __init__(self):
        self.records = []
        self.result = []
    
    def findOptimalEnhancement(self, image, methode, parameter_space):
        original = image.astype(np.float64)

        sigma, clip_limits, median_sizes = parameter_space
        
        records = gaussian(original, )
        
        df = pd.DataFrame(records)

        df_sorted = df.sort_values(by="PSNR", ascending=False)

        print(df_sorted.head(10))

        top_n = 6
        top_indices = df_sorted.head(top_n).index

        plt.figure(figsize=(12,6))
        for i, idx in enumerate(top_indices):
            name, img = results[idx]
            plt.subplot(2, 3, i+1)
            plt.imshow(img, cmap='gray')
            plt.title(name)
            plt.axis('off')

        plt.suptitle("Top Enhancement Results (by PSNR)")
        plt.tight_layout()
        plt.show()

    def gaussian(self, im=None, sigma=1, eq=False, clip_limits=0.01):
        filtering = ndi.gaussian_filter(im, sigma=sigma)
        if eq == True and clip_limits > 0:
            enhanced = exposure.equalize_adapthist(filtering, clip_limits=clip_limits)
        elif eq == True and clip_limits == None:
            enhanced = exposure.equalize_hist(filtering)
            
        psnr, mse, nm = AnalysisPreROI("Gaussian", im, filtering)
        
        return self.records.append({
            "Method": "Gaussian+CLAHE",
            "Sigma": sigma,
            "Median": None,
            "MSE": mse,
            "PSNR": psnr
        })
        
        
