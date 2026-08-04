"""Evaluate the best checkpoint on the held-out test set.

Reports per-class IoU and Dice (not pixel accuracy — most pixels are
background, so accuracy would look great on a useless model).

Writes outputs/metrics.json — the SOLE source for any number in the README
or portfolio site.

Usage: python src/evaluate.py
"""
import json
import sys
import time
from pathlib import Path

import segmentation_models_pytorch as smp  # noqa: F401 (model built via train)
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src.dataset import TileDataset, eval_transforms
from src.train import build_model, get_device


@torch.no_grad()
def main() -> None:
    device = get_device()
    ckpt_path = config.CKPT_DIR / "best.pt"
    if not ckpt_path.exists():
        sys.exit("No best.pt checkpoint. Train first.")
    state = torch.load(ckpt_path, map_location=device)
    model = build_model().to(device)
    model.load_state_dict(state["model"])
    model.eval()

    ds = TileDataset("test", eval_transforms())
    dl = DataLoader(ds, batch_size=config.BATCH_SIZE.get(device, 4),
                    shuffle=False, num_workers=config.NUM_WORKERS)

    C = config.NUM_CLASSES
    cm = torch.zeros(C, C, dtype=torch.long)
    times = []
    for imgs, masks, _ in dl:
        imgs = imgs.to(device)
        t0 = time.perf_counter()
        logits = model(imgs)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) / imgs.size(0))
        pred = logits.argmax(1).cpu()
        idx = masks.reshape(-1) * C + pred.reshape(-1)
        cm += torch.bincount(idx, minlength=C * C).reshape(C, C)

    tp = cm.diag().float()
    fp = cm.sum(0).float() - tp
    fn = cm.sum(1).float() - tp
    iou = (tp / (tp + fp + fn).clamp(min=1)).tolist()
    dice = ((2 * tp) / (2 * tp + fp + fn).clamp(min=1)).tolist()
    support = cm.sum(1).tolist()  # true pixel count per class

    metrics = {
        "checkpoint": str(ckpt_path),
        "checkpoint_epoch": state.get("epoch"),
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "device": device,
        "n_test_tiles": len(ds),
        "class_names": config.CLASS_NAMES,
        "iou_per_class": {n: round(v, 4) for n, v in zip(config.CLASS_NAMES, iou)},
        "dice_per_class": {n: round(v, 4) for n, v in zip(config.CLASS_NAMES, dice)},
        "mean_iou": round(sum(iou) / len(iou), 4),
        "mean_dice": round(sum(dice) / len(dice), 4),
        "test_pixel_support": {n: int(s) for n, s in zip(config.CLASS_NAMES, support)},
        "mean_inference_ms_per_tile": round(1000 * sum(times) / len(times), 2),
        "confusion_matrix_rows_true_cols_pred": cm.tolist(),
    }
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.OUTPUTS_DIR / "metrics.json"
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print("\nWritten to", out)


if __name__ == "__main__":
    main()
