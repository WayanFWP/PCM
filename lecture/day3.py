import pydicom as pydcm
import numpy as np
import matplotlib.pyplot as plt

sample_dcm = "dataset/dicom/dicom_dir/ID_0000_AGE_0060_CONTRAST_1_CT.dcm"
dicom_file = pydcm.read_file(sample_dcm)
# print(dicom_file)

ct = dicom_file.pixel_array
plt.figure()
plt.imshow(ct, cmap="gray")
plt.show()

print(ct.shape)
