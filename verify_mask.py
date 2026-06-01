from PIL import Image
import numpy as np

p = r"train/masks_aligned/119_sat.png"
arr = np.array(Image.open(p))
print("Unique colors:", np.unique(arr.reshape(-1,3), axis=0)[:10])
