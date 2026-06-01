# align_masks.py
import os, shutil
from glob import glob

def align(images_dir, masks_clean_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    images = sorted([f for f in os.listdir(images_dir) if f.lower().endswith((".jpg",".jpeg",".png",".tif"))])
    masks = sorted([f for f in os.listdir(masks_clean_dir) if f.lower().endswith(".png")])
    mask_index = {}
    for m in masks:
        key = m.split('_')[0]
        mask_index.setdefault(key, []).append(m)
    copied = 0
    missed = []
    for img in images:
        key = img.split('_')[0]
        candidates = mask_index.get(key, [])
        if candidates:
            src = os.path.join(masks_clean_dir, candidates[0])
            tgt_name = os.path.splitext(img)[0] + ".png"
            tgt = os.path.join(out_dir, tgt_name)
            shutil.copy(src, tgt)
            copied += 1
        else:
            missed.append(img)
    print(f"Aligned {copied} masks; missed {len(missed)} images")
    if missed:
        print("Sample missed images:", missed[:10])

if __name__ == '__main__':
    align("train/images", "train/masks_clean", "train/masks_aligned")
    align("valid/images", "valid/masks_clean", "valid/masks_aligned")
