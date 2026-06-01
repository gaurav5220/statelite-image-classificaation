# my_dataset.py
import os
from glob import glob
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random

class SegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=(256,256), split="train"):
        self.img_paths = sorted(glob(os.path.join(img_dir, "*")))
        self.mask_paths = sorted(glob(os.path.join(mask_dir, "*")))
        assert len(self.img_paths) == len(self.mask_paths), f"Image/Mask count mismatch: {len(self.img_paths)} vs {len(self.mask_paths)}"
        self.img_size = img_size
        self.split = split

        # palette expected
        self.palette = {
            (255,255,0): 0,
            (255,0,255): 1,
            (0,0,255): 2,
            (0,255,255): 3
        }
        self._pal_arr = np.array(list(self.palette.keys())).astype(np.int32)

        self.to_tensor = transforms.ToTensor()

        # mild augmentations for images, flips-only for masks
        self.aug_img = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.05, hue=0.01)
        ])
        self.aug_mask = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip()
        ])

        self.resize_img = transforms.Resize(self.img_size, interpolation=Image.BILINEAR)
        self.resize_mask = transforms.Resize(self.img_size, interpolation=Image.NEAREST)

    def __len__(self):
        return len(self.img_paths)

    def rgb_to_class(self, mask_np):
        if mask_np.ndim == 2:
            return mask_np.astype(np.int64)
        h,w,_ = mask_np.shape
        out = np.zeros((h,w), dtype=np.int64)
        for rgb, cid in self.palette.items():
            out[(mask_np == rgb).all(axis=-1)] = cid
        uniq = np.unique(out)
        if len(uniq) <= 1:
            flat = mask_np.reshape(-1,3).astype(np.int32)
            d = ((flat[:,None,:] - self._pal_arr[None,:,:])**2).sum(axis=2)
            idx = d.argmin(axis=1).reshape(h,w)
            return idx.astype(np.int64)
        return out

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("RGB")

        if self.split == "train":
            seed = random.randint(0, 999999)
            random.seed(seed); img = self.aug_img(img)
            random.seed(seed); mask = self.aug_mask(mask)

        img = self.resize_img(img)
        mask = self.resize_mask(mask)

        img_t = self.to_tensor(img)
        mask_np = np.array(mask)
        mask_class = self.rgb_to_class(mask_np)
        mask_t = torch.tensor(mask_class, dtype=torch.long)

        return img_t, mask_t
