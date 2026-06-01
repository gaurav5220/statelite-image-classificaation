from my_dataset import SegDataset

ds = SegDataset("train/images", "train/masks_aligned")
img, mask = ds[0]

print("IMG:", img.shape)
print("MASK:", mask.shape)
