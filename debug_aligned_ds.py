from my_dataset import SegDataset

train_ds = SegDataset("train/images", "train/masks_aligned", img_size=(256,256), split="train")

print("Train aligned length:", len(train_ds))

for i in range(5):
    img, mask = train_ds[i]
    print(f"Sample {i}: IMG shape={img.shape} MASK unique={mask.unique().tolist()}")
