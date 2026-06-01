import os

imgs = sorted(os.listdir("train/images"))[:20]
masks = sorted(os.listdir("train/masks_aligned"))[:20]

for i in range(20):
    print(f"{i}: IMG={imgs[i]}   MASK={masks[i]}")
