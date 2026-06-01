from PIL import Image
import numpy as np

import os

f = "train/masks_aligned/100694_sat.png"
mask = Image.open(f)               # original
arr1 = np.array(mask)
print("ORIGINAL unique:", np.unique(arr1))

res = mask.resize((256,256), Image.NEAREST)
arr2 = np.array(res)
print("RESIZED unique:", np.unique(arr2))
