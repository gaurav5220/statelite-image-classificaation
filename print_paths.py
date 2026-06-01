from my_dataset import SegDataset
ds = SegDataset("train/images", "train/masks_aligned", img_size=(256,256))
print("First mask path:", ds.mask_paths[0])
