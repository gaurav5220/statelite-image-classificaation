from my_dataset import SegDataset

ds = SegDataset("train/images", "train/masks", img_size=(256,256))
img, mask = ds[0]

print("IMAGE SHAPE:", img.shape)
print("MASK SHAPE:", mask.shape)
