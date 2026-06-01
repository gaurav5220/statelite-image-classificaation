from PIL import Image
import numpy as np
import os

folder = "train/masks_aligned"
files = sorted([f for f in os.listdir(folder)])[:10]

for f in files:
    arr = np.array(Image.open(os.path.join(folder,f)))
    print(f, "unique:", np.unique(arr)[:10], "(count:", len(np.unique(arr)),")")
