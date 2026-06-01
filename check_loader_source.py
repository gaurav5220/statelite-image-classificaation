from my_dataset import SegDataset

ds = SegDataset("train/images", "train/masks_aligned", img_size=(256,256))

print("Loaded MASK path 0:", ds.mask_paths[0])

img, mask = ds[0]
print("Mask unique from dataset:", mask.unique().tolist())
