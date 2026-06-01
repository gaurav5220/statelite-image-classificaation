from my_dataset import SegDataset

ds = SegDataset("train/images", "train/masks", img_size=(256,256))
print("Dataset length:", len(ds))

# show first 3 samples info (shapes & dtype & min/max)
for i in range(min(3, len(ds))):
    img, mask = ds[i]
    print(f"Sample {i}: IMG shape={tuple(img.shape)} dtype={img.dtype} minmax=({img.min().item()},{img.max().item()})")
    print(f"Sample {i}: MASK shape={tuple(mask.shape)} dtype={mask.dtype} unique={sorted(mask.unique().tolist())}")
