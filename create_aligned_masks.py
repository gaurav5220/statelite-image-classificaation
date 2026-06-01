import os
from PIL import Image

src_train = "train/masks"
src_valid = "valid/masks"

dst_train = "train/masks_aligned"
dst_valid = "valid/masks_aligned"

os.makedirs(dst_train, exist_ok=True)
os.makedirs(dst_valid, exist_ok=True)

def make_aligned(src, dst):
    files = sorted(os.listdir(src))
    count = 0
    for f in files:
        if not f.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        img = Image.open(os.path.join(src, f)).convert("RGB")
        save_name = f.replace("_mask", "_sat")
        img.save(os.path.join(dst, save_name))
        count += 1

    print(f"[OK] {count} masks saved -> {dst}")

make_aligned(src_train, dst_train)
make_aligned(src_valid, dst_valid)
