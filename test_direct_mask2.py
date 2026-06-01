from PIL import Image
import numpy as np

path = r"train/masks_aligned/100694_sat.png"
arr = np.array(Image.open(path).convert("L"))
print("Unique values in THIS file:", np.unique(arr))
