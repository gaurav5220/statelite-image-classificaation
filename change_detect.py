
import os
import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
from my_dataset import SegDataset

from torch.utils.data import DataLoader
from PIL import Image

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def load_filenames(img_dir, mask_dir):
    imgs = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if not f.startswith('.')])
    masks = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if not f.startswith('.')])
    return imgs, masks

def open_image_as_numpy(path, img_size=None):
    img = Image.open(path).convert("RGB")
    if img_size is not None:
        img = img.resize(img_size, Image.BILINEAR)
    return np.array(img)

def open_mask_as_numpy(path, img_size=None):
    m = Image.open(path).convert("RGB")
    if img_size is not None:
        m = m.resize(img_size, Image.NEAREST)
    return np.array(m)

def rgb_to_class_mask_from_dataset(mask_np, class_colors):
    """
    Convert RGB mask (H,W,3) -> class indices (H,W) using class_colors mapping.
    class_colors: dict {(R,G,B): class_id}
    If class_colors is None, fallback to using red channel.
    """
    if class_colors is None:
        return mask_np[:, :, 0].astype(np.int64)

    h, w, _ = mask_np.shape
    out = np.zeros((h, w), dtype=np.int64)
    for rgb, cls in class_colors.items():
        match = np.all(mask_np == rgb, axis=-1)
        out[match] = cls
    return out

def compute_change_stats(mask_old_cls, mask_new_cls, num_classes):
    """
    Returns: total_changed_pixels, percent_changed, per_class_changes dict
    per_class_changes: {class_id: {'from_to': {to_class: count}, 'changed_out': count, 'changed_in': count}}
    """
    total_pixels = mask_old_cls.size
    changed_mask = (mask_old_cls != mask_new_cls)
    changed_pixels = changed_mask.sum()
    percent_changed = 100.0 * changed_pixels / total_pixels

    per_class = {}
    for c in range(num_classes):
        per_class[c] = {
            'from_count': int((mask_old_cls == c).sum()),
            'to_count': int((mask_new_cls == c).sum()),
            'changed_out': int(((mask_old_cls == c) & (mask_new_cls != c)).sum()),
            'changed_in': int(((mask_old_cls != c) & (mask_new_cls == c)).sum()),
        }

    # detailed transition counts
    transitions = {}
    for a in range(num_classes):
        transitions[a] = {}
        for b in range(num_classes):
            count = int(((mask_old_cls == a) & (mask_new_cls == b)).sum())
            transitions[a][b] = count

    return int(changed_pixels), percent_changed, per_class, transitions

def visualize_and_save(idx, img_old_np, img_new_np, mask_old_cls, mask_new_cls, class_color_map, class_name_map, outdir):
    """
    Save a combined figure with:
    [old image] [new image] [old mask colored] [new mask colored] [change highlight overlay on new image]
    """
    h, w = mask_old_cls.shape
    # map class ids to colors
    mask_old_color = np.zeros((h,w,3), dtype=np.uint8)
    mask_new_color = np.zeros((h,w,3), dtype=np.uint8)
    for cid, col in class_color_map.items():
        mask_old_color[mask_old_cls == cid] = col
        mask_new_color[mask_new_cls == cid] = col

    # change overlay: red where different, overlay on top of new image for context
    change_mask = (mask_old_cls != mask_new_cls)
    overlay = img_new_np.copy()
    overlay[change_mask] = (255, 0, 0)  # red highlight

    # compose figure
    plt.figure(figsize=(18,6))

    plt.subplot(1,5,1)
    plt.imshow(img_old_np)
    plt.title("OLD Image")
    plt.axis('off')

    plt.subplot(1,5,2)
    plt.imshow(img_new_np)
    plt.title("NEW Image")
    plt.axis('off')

    plt.subplot(1,5,3)
    plt.imshow(mask_old_color)
    plt.title("OLD Mask")
    plt.axis('off')

    plt.subplot(1,5,4)
    plt.imshow(mask_new_color)
    plt.title("NEW Mask")
    plt.axis('off')

    plt.subplot(1,5,5)
    plt.imshow(overlay)
    plt.title("CHANGES (red)")
    plt.axis('off')

    out_path = os.path.join(outdir, f"change_{idx}.png")
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    return out_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--old_images', type=str, default="train/images")
    parser.add_argument('--old_masks', type=str, default="train/masks")
    parser.add_argument('--new_images', type=str, default="valid/images")
    parser.add_argument('--new_masks', type=str, default="valid/masks")
    parser.add_argument('--img_size', type=int, default=256, help="visualization resize (square)")
    parser.add_argument('--max_pairs', type=int, default=0, help="max pairs to process (0 = all)")
    parser.add_argument('--outdir', type=str, default="change_out")
    parser.add_argument('--num_classes', type=int, default=4, help="number of classes")
    args = parser.parse_args()

    ensure_dir(args.outdir)

    # Load file lists (sorted)
    old_imgs, old_masks = load_filenames(args.old_images, args.old_masks)
    new_imgs, new_masks = load_filenames(args.new_images, args.new_masks)

    n_pairs = min(len(old_imgs), len(new_imgs), len(old_masks), len(new_masks))
    if args.max_pairs and args.max_pairs > 0:
        n_pairs = min(n_pairs, args.max_pairs)
    print(f"Found {n_pairs} pairs to process.")

    # Use dataset's class mapping if available by instantiating SegDataset and reading its class_colors attribute.
    # We assume the dataset.py you use has same class_colors mapping as earlier.
    # Fallback: use simple mapping of unique colors found in masks -> indices (not ideal).
    try:
        ds_tmp = SegDataset(args.old_images, args.old_masks, img_size=(args.img_size, args.img_size))
        class_colors = getattr(ds_tmp, "class_colors", None)
        print("Using class_colors from SegDataset.")
    except Exception as e:
        print("Warning: could not instantiate SegDataset to read mapping:", e)
        class_colors = None

    # Build color map for visualization: id -> RGB
    if class_colors is not None:
        id_to_color = {v: k for k, v in class_colors.items()}
    else:
        # fallback: create incremental colors (not ideal)
        id_to_color = {i: (int(255*(i%3==0)), int(255*(i%3==1)), int(255*(i%3==2))) for i in range(args.num_classes)}

    # Name map (optional)
    id_to_name = {i: f"Class_{i}" for i in range(args.num_classes)}

    # CSV summary
    csv_path = os.path.join(args.outdir, "change_summary.csv")
    csv_fields = ["pair_index","old_img","new_img","changed_pixels","percent_changed"]
    for c in range(args.num_classes):
        csv_fields += [f"class_{c}_from", f"class_{c}_to", f"class_{c}_changed_out", f"class_{c}_changed_in"]
    csv_file = open(csv_path, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    writer.writeheader()

    for i in range(n_pairs):
        old_img_path = old_imgs[i]
        new_img_path = new_imgs[i]
        old_mask_path = old_masks[i]
        new_mask_path = new_masks[i]

        img_old_np = open_image_as_numpy(old_img_path, img_size=(args.img_size, args.img_size))
        img_new_np = open_image_as_numpy(new_img_path, img_size=(args.img_size, args.img_size))
        mask_old_np = open_mask_as_numpy(old_mask_path, img_size=(args.img_size, args.img_size))
        mask_new_np = open_mask_as_numpy(new_mask_path, img_size=(args.img_size, args.img_size))

        # convert rgb->class indices
        mask_old_cls = rgb_to_class_mask_from_dataset(mask_old_np, class_colors)
        mask_new_cls = rgb_to_class_mask_from_dataset(mask_new_np, class_colors)

        changed_pixels, percent_changed, per_class, transitions = compute_change_stats(mask_old_cls, mask_new_cls, args.num_classes)

        # visualize and save
        out_img_path = visualize_and_save(i, img_old_np, img_new_np, mask_old_cls, mask_new_cls, id_to_color, id_to_name, args.outdir)

        # write csv row
        row = {
            "pair_index": i,
            "old_img": os.path.basename(old_img_path),
            "new_img": os.path.basename(new_img_path),
            "changed_pixels": changed_pixels,
            "percent_changed": round(percent_changed, 4)
        }
        for c in range(args.num_classes):
            row[f"class_{c}_from"] = per_class[c]['from_count']
            row[f"class_{c}_to"] = per_class[c]['to_count']
            row[f"class_{c}_changed_out"] = per_class[c]['changed_out']
            row[f"class_{c}_changed_in"] = per_class[c]['changed_in']

        writer.writerow(row)
        print(f"[{i}] {os.path.basename(old_img_path)} -> {os.path.basename(new_img_path)} | changed {changed_pixels} pixels ({percent_changed:.3f}%) | saved {out_img_path}")

    csv_file.close()
    print("All done. CSV summary at:", csv_path)

if __name__ == "__main__":
    main()
