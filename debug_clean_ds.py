from my_dataset import SegDataset

ds = SegDataset("train/images", "train/masks_clean", img_size=(256,256))
print("Dataset length:", len(ds))

for i in range(5):
    img, mask = ds[i]
    print(f"Sample {i}: IMG shape={tuple(img.shape)} dtype={img.dtype} minmax=({img.min().item()},{img.max().item()})")
    print(f"Sample {i}: MASK shape={tuple(mask.shape)} dtype={mask.dtype} unique={mask.unique().tolist()}")
