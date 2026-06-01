from PIL import Image
import numpy as np

paths = [
    r'train/masks_aligned/100694_sat.png',
    r'train/masks_aligned/102122_sat.png',
    r'train/masks_aligned/103665_sat.png'
]

for p in paths:
    arr = np.array(Image.open(p))
    print(p, 'unique RGB:', np.unique(arr.reshape(-1,3), axis=0)[:20])
