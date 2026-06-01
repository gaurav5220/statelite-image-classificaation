# inspect_masks.py  (robust, prints details)
import os
from PIL import Image
from collections import Counter

def list_image_files(folder):
    exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(exts)])

def show_colors(folder, max_files=50):
    files = list_image_files(folder)
    print(f"\nFolder: {folder}")
    print("Total files found:", len(files))
    if len(files) == 0:
        return
    color_counter = Counter()
    inspected = 0
    for fname in files[:max_files]:
        path = os.path.join(folder, fname)
        try:
            im = Image.open(path).convert("RGB")
        except Exception as e:
            print("  [ERROR opening]", path, e)
            continue
        small = im.resize((128,128))
        colors = small.getcolors(maxcolors=1000000)
        if colors:
            for cnt, col in colors:
                color_counter[col] += 1
        inspected += 1

    most = color_counter.most_common(30)
    print(f"Files inspected (sampled): {inspected}")
    if not most:
        print("  No colors found (strange).")
    else:
        print("Top colors (rgb -> occurrences among sampled files):")
        for col, ct in most:
            print(f"  {col} -> {ct}")

if __name__ == "__main__":
    for folder in ["train/masks", "valid/masks"]:
        if os.path.isdir(folder):
            show_colors(folder, max_files=50)
        else:
            print(f"Missing folder: {folder}")
    print("\nDone.")
