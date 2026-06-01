from my_dataset import SegDataset
import os

ds = SegDataset("train/images", "train/masks_aligned")
full_paths = [os.path.abspath(p) for p in ds.mask_paths[:5]]
for p in full_paths:
    print(p)
