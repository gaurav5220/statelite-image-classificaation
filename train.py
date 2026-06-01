# train.py
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from my_dataset import SegDataset

# ---------------- METRICS ----------------
def pixel_accuracy(pred_logits, target):
    pred_labels = pred_logits.argmax(dim=1)
    correct = (pred_labels == target).sum().item()
    total = target.numel()
    return correct / total if total > 0 else 0.0

def intersection_over_union(pred_logits, target, num_classes):
    pred = pred_logits.argmax(dim=1).view(-1)
    tgt = target.view(-1)
    eps = 1e-6
    ious = []
    for c in range(num_classes):
        p = (pred == c)
        t = (tgt == c)
        inter = (p & t).sum().item()
        union = (p | t).sum().item()
        if union > 0:
            ious.append(inter / (union + eps))
    return float(np.mean(ious)) if ious else 0.0

# ---------------- EPOCHS ----------------
def train_epoch(model, loader, optimizer, device, num_classes, loss_fn, max_batches=None, use_tq=True):
    model.train()
    total_l = 0.0
    total_iou = 0.0
    total_acc = 0.0
    n_batches = 0
    iterator = tqdm(loader, desc="Training") if use_tq else loader
    for batch_idx, (imgs, masks) in enumerate(iterator):
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        out = out["out"] if isinstance(out, dict) and "out" in out else out
        loss = loss_fn(out, masks)
        loss.backward()
        optimizer.step()

        total_l += loss.item()
        total_acc += pixel_accuracy(out.detach(), masks)
        total_iou += intersection_over_union(out.detach(), masks, num_classes)
        n_batches += 1

        if max_batches and n_batches >= max_batches:
            break

    if n_batches == 0:
        return 0.0, 0.0, 0.0
    return total_l / n_batches, total_iou / n_batches, total_acc / n_batches

def val_epoch(model, loader, device, num_classes, loss_fn, max_batches=None, use_tq=True):
    model.eval()
    total_l = 0.0
    total_iou = 0.0
    total_acc = 0.0
    n_batches = 0
    iterator = tqdm(loader, desc="Validation") if use_tq else loader
    with torch.no_grad():
        for batch_idx, (imgs, masks) in enumerate(iterator):
            imgs, masks = imgs.to(device), masks.to(device)
            out = model(imgs)
            out = out["out"] if isinstance(out, dict) and "out" in out else out
            loss = loss_fn(out, masks)
            total_l += loss.item()
            total_acc += pixel_accuracy(out, masks)
            total_iou += intersection_over_union(out, masks, num_classes)
            n_batches += 1
            if max_batches and n_batches >= max_batches:
                break

    if n_batches == 0:
        return 0.0, 0.0, 0.0
    return total_l / n_batches, total_iou / n_batches, total_acc / n_batches

# ----------------- MAIN -----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--num_classes', type=int, required=True)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--max_batches', type=int, default=0, help='0=no limit')
    parser.add_argument('--no_tqdm', action='store_true')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--test_only', action='store_true')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    print("Args:", args)

    train_img = "train/images"
    train_mask = "train/masks_aligned"
    valid_img = "valid/images"
    valid_mask = "valid/masks_aligned"

    train_ds = SegDataset(train_img, train_mask, img_size=(args.img_size, args.img_size), split="train")
    val_ds = SegDataset(valid_img, valid_mask, img_size=(args.img_size, args.img_size), split="val")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, drop_last=False)

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    # --- model: use pretrained weights for better convergence ---
    try:
        weights_enum = models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
        model = models.segmentation.deeplabv3_resnet50(weights=weights_enum)
    except Exception:
        model = models.segmentation.deeplabv3_resnet50(weights=None)
        print("Warning: pretrained weights not available, using weights=None")

    # replace classifier output channels
    try:
        model.classifier[-1] = nn.Conv2d(model.classifier[-1].in_channels, args.num_classes, kernel_size=1)
    except Exception:
        model.classifier = nn.Conv2d(256, args.num_classes, 1)

    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    # resume checkpoint if provided
    start_epoch = 1
    if args.resume and os.path.isfile(args.resume):
        state = torch.load(args.resume, map_location=device)
        if isinstance(state, dict) and 'model_state' in state:
            model.load_state_dict(state['model_state'])
            if 'optimizer_state' in state:
                try:
                    optimizer.load_state_dict(state['optimizer_state'])
                except Exception:
                    print("Warning: optimizer state load failed.")
            start_epoch = state.get('epoch', 0) + 1
        else:
            model.load_state_dict(state)
            start_epoch = 1
        print("Resumed from", args.resume, "starting epoch", start_epoch)

    os.makedirs("checkpoints", exist_ok=True)

    if args.test_only:
        model.eval()
        os.makedirs("preds_vis", exist_ok=True)
        with torch.no_grad():
            for i, (imgs, masks) in enumerate(val_loader):
                imgs = imgs.to(device)
                out = model(imgs)
                preds = out["out"] if isinstance(out, dict) and "out" in out else out
                pred_labels = preds.argmax(dim=1).cpu().numpy()
                imgs_np = imgs.cpu().numpy()
                masks_np = masks.numpy()
                for b in range(pred_labels.shape[0]):
                    img = (imgs_np[b].transpose(1,2,0)*255).astype('uint8')
                    gt = masks_np[b]
                    pred = pred_labels[b]
                    change_mask = (gt != pred)
                    change_vis = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)
                    change_vis[change_mask] = [255,0,0]
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(1,4,figsize=(14,4))
                    ax[0].imshow(img); ax[0].axis('off'); ax[0].set_title("Img")
                    ax[1].imshow(gt); ax[1].axis('off'); ax[1].set_title("GT")
                    ax[2].imshow(pred); ax[2].axis('off'); ax[2].set_title("Pred")
                    ax[3].imshow(change_vis); ax[3].axis('off'); ax[3].set_title("Change")
                    plt.savefig(f"preds_vis/vis_{i}_{b}.png", bbox_inches='tight', pad_inches=0)
                    plt.close(fig)
        return

    # training loop
    use_tq = not args.no_tqdm
    max_b = args.max_batches if args.max_batches > 0 else None

    train_losses, val_losses = [], []
    train_ious, val_ious = [], []
    train_accs, val_accs = [], []

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        tr_loss, tr_iou, tr_acc = train_epoch(model, train_loader, optimizer, device, args.num_classes, loss_fn, max_batches=max_b, use_tq=use_tq)
        va_loss, va_iou, va_acc = val_epoch(model, val_loader, device, args.num_classes, loss_fn, max_batches=max_b, use_tq=use_tq)

        print(f"Train Loss={tr_loss:.4f} IoU={tr_iou:.4f} Acc={tr_acc:.4f}")
        print(f"Val   Loss={va_loss:.4f} IoU={va_iou:.4f} Acc={va_acc:.4f}")

        train_losses.append(tr_loss); val_losses.append(va_loss)
        train_ious.append(tr_iou); val_ious.append(va_iou)
        train_accs.append(tr_acc); val_accs.append(va_acc)

        ckpt = {"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict()}
        torch.save(ckpt, os.path.join("checkpoints", f"ckpt_epoch_{epoch}.pth"))
        print("Saved checkpoint:", os.path.join("checkpoints", f"ckpt_epoch_{epoch}.pth"))

        scheduler.step(va_loss)

    # save plots
    try:
        plt.figure(); plt.plot(train_losses, label="train"); plt.plot(val_losses, label="val"); plt.legend(); plt.savefig("loss_curve.png"); plt.close()
        plt.figure(); plt.plot(train_ious, label="train"); plt.plot(val_ious, label="val"); plt.legend(); plt.savefig("iou_curve.png"); plt.close()
        plt.figure(); plt.plot(train_accs, label="train"); plt.plot(val_accs, label="val"); plt.legend(); plt.savefig("acc_curve.png"); plt.close()
        print("Saved plots: loss_curve.png, iou_curve.png, acc_curve.png")
    except Exception:
        pass

if __name__ == "__main__":
    main()
