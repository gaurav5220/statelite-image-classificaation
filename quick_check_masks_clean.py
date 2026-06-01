# quick_check_masks_clean.py
import os
from PIL import Image
import numpy as np

def show(folder, n=8):
    files = sorted([f for f in os.listdir(folder) if f.lower().endswith('.png')])
    print("Folder:", folder, " total:", len(files))
    for f in files[:n]:
        p = os.path.join(folder, f)
        im = Image.open(p)
        arr = np.array(im)
        uniq = np.unique(arr)
        print(f" {f}  unique_values: {uniq[:10]} (count {len(uniq)})")
    print()

for fld in ["train/masks_clean", "valid/masks_clean"]:
    if os.path.isdir(fld):
        show(fld, n=8)
    else:
        print("Missing folder:", fld)
