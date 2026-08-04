"""Train U-Net (ResNet34 encoder, ImageNet-pretrained) on the prepared tiles.

Loss = Dice + cross-entropy. Adam @ 1e-4. Early-stops when val loss plateaus.
Logs every epoch to outputs/training_log.csv — that file is the evidence.

Usage:
  python src/train.py --smoke     # 2 epochs on 64 tiles; verifies the pipeline
  python src/train.py             # real run
  python src/train.py --resume    # continue from outputs/checkpoints/last.pt
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src.dataset import TileDataset, eval_transforms, train_transforms


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_model():
    return smp.Unet(
        encoder_name=config.ENCODER,
        encoder_weights=config.ENCODER_WEIGHTS,
        in_channels=3,
        classes=config.NUM_CLASSES,
    )


def miou(cm: torch.Tensor) -> float:
    tp = cm.diag().float()
    iou = tp / (cm.sum(0).float() + cm.sum(1).float() - tp).clamp(min=1)
    return iou.mean().item()


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()
    total, n = 0.0, 0
    cm = torch.zeros(config.NUM_CLASSES, config.NUM_CLASSES, dtype=torch.long)
    for imgs, masks, _ in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        total += loss_fn(logits, masks).item() * imgs.size(0)
        n += imgs.size(0)
        pred = logits.argmax(1)
        idx = masks.reshape(-1).cpu() * config.NUM_CLASSES + pred.reshape(-1).cpu()
        cm += torch.bincount(idx, minlength=config.NUM_CLASSES ** 2).reshape(
            config.NUM_CLASSES, config.NUM_CLASSES)
    return total / max(n, 1), miou(cm)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="tiny 2-epoch run to verify the pipeline end to end")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--epochs", type=int, default=config.MAX_EPOCHS)
    args = ap.parse_args()

    torch.manual_seed(config.SEED)
    device = get_device()
    bs = config.BATCH_SIZE.get(device, 4)
    workers = 0 if args.smoke else config.NUM_WORKERS
    print(f"device={device} batch_size={bs}")

    train_ds = TileDataset("train", train_transforms())
    val_ds = TileDataset("val", eval_transforms())
    if args.smoke:
        train_ds = Subset(train_ds, range(min(64, len(train_ds))))
        val_ds = Subset(val_ds, range(min(32, len(val_ds))))
        args.epochs = 2

    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True,
                          num_workers=workers, pin_memory=(device == "cuda"))
    val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False,
                        num_workers=workers, pin_memory=(device == "cuda"))

    model = build_model().to(device)
    dice = smp.losses.DiceLoss(mode="multiclass", from_logits=True)
    ce = smp.losses.SoftCrossEntropyLoss(smooth_factor=0.1)

    def loss_fn(logits, masks):
        # Dice optimizes region overlap directly (robust to class imbalance);
        # CE gives smooth per-pixel gradients. Equal weights — a starting
        # point, not a tuned magic number.
        return dice(logits, masks) + ce(logits, masks)

    opt = torch.optim.Adam(model.parameters(), lr=config.LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=0.5, patience=config.PLATEAU_PATIENCE)

    config.CKPT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = config.OUTPUTS_DIR / "training_log.csv"
    start_epoch, best_val, since_best = 1, float("inf"), 0

    if args.resume and (config.CKPT_DIR / "last.pt").exists():
        state = torch.load(config.CKPT_DIR / "last.pt", map_location=device)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        start_epoch = state["epoch"] + 1
        best_val = state["best_val"]
        since_best = state["since_best"]
        print(f"Resumed from epoch {state['epoch']}")
    elif not args.smoke and log_path.exists() and not args.resume:
        log_path.unlink()  # fresh run, fresh log

    if not log_path.exists():
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["epoch", "train_loss", "val_loss", "val_miou", "lr", "seconds"])

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        model.train()
        total, n = 0.0, 0
        for imgs, masks, _ in train_dl:
            imgs, masks = imgs.to(device), masks.to(device)
            opt.zero_grad()
            loss = loss_fn(model(imgs), masks)
            loss.backward()
            opt.step()
            total += loss.item() * imgs.size(0)
            n += imgs.size(0)
        train_loss = total / max(n, 1)
        val_loss, val_miou = validate(model, val_dl, loss_fn, device)
        sched.step(val_loss)
        lr_now = opt.param_groups[0]["lr"]
        secs = time.time() - t0
        print(f"epoch {epoch:3d}  train {train_loss:.4f}  val {val_loss:.4f}"
              f"  val_mIoU {val_miou:.4f}  lr {lr_now:.2e}  {secs:.0f}s")
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, f"{train_loss:.5f}", f"{val_loss:.5f}",
                 f"{val_miou:.5f}", f"{lr_now:.2e}", f"{secs:.1f}"])

        if val_loss < best_val:
            best_val, since_best = val_loss, 0
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_loss": val_loss, "val_miou": val_miou},
                       config.CKPT_DIR / "best.pt")
        else:
            since_best += 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "epoch": epoch, "best_val": best_val,
                    "since_best": since_best}, config.CKPT_DIR / "last.pt")
        if not args.smoke and since_best >= config.EARLY_STOP_PATIENCE:
            print(f"Early stop: no val improvement in {since_best} epochs.")
            break

    summary = {"device": device, "batch_size": bs, "best_val_loss": best_val,
               "epochs_run": epoch, "smoke": args.smoke,
               "encoder": config.ENCODER, "lr_start": config.LR,
               "date": time.strftime("%Y-%m-%d %H:%M")}
    with open(config.OUTPUTS_DIR / "train_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Done. Log:", log_path)


if __name__ == "__main__":
    main()
