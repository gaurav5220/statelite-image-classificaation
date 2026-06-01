# find_black_valid_masks.py
import os
from PIL import Image

folder = "valid/masks"
files = sorted([f for f in os.listdir(folder) if f.lower().endswith(('.png','.jpg','.jpeg','.tif'))])
bad = []
for f in files:
    p = os.path.join(folder, f)
    try:
        im = Image.open(p).convert("RGB")
    except:
        bad.append((f, "open_error"))
        continue
    colors = im.resize((64,64)).getcolors(maxcolors=1000000)
    if not colors:
        bad.append((f, "no_colors"))
    elif len(colors)==1:
        bad.append((f, colors[0][1]))
# print summary
print("Total valid masks:", len(files))
print("Single-color (likely bad) masks:", len(bad))
print("Sample bad files (first 20):")
for item in bad[:20]:
    print(" ", item)
