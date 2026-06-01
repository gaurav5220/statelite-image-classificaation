# clean_masks.py
import os, numpy as np
from PIL import Image
from tqdm import tqdm

PALETTE = {
    (255,255,0): 0,   # agriculture
    (255,0,255): 1,   # urban
    (0,0,255): 2,     # water
    (0,255,255): 3    # forest
}
pal_arr = np.array(list(PALETTE.keys())).astype(np.int32)

def rgb_to_nearest(mask_np):
    h,w,_ = mask_np.shape
    flat = mask_np.reshape(-1,3).astype(np.int32)
    d = ((flat[:,None,:] - pal_arr[None,:,:])**2).sum(axis=2)
    idx = d.argmin(axis=1)
    class_mask = idx.reshape(h,w).astype(np.uint8)
    return class_mask

def process_folder(src_folder, dst_folder):
    os.makedirs(dst_folder, exist_ok=True)
    files = sorted([f for f in os.listdir(src_folder) if f.lower().endswith(('.png','.jpg','.jpeg','.tif','.tiff'))])
    for f in tqdm(files):
        p = os.path.join(src_folder, f)
        try:
            im = Image.open(p).convert("RGB")
        except Exception as e:
            print("Error open:", p, e)
            continue
        arr = np.array(im)
        class_mask = rgb_to_nearest(arr)
        outp = os.path.join(dst_folder, os.path.splitext(f)[0] + "_clean.png")
        Image.fromarray(class_mask).save(outp)

if __name__ == "__main__":
    print("Cleaning train masks -> train/masks_clean")
    process_folder("train/masks", "train/masks_clean")
    print("Cleaning valid masks -> valid/masks_clean")
    process_folder("valid/masks", "valid/masks_clean")
    print("Done.")
