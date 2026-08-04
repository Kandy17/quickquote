"""Inference on test tiles -> class map -> polygon boundaries -> overlays.

This is the bridge from model output to a highlightable region: OpenCV
findContours turns the per-pixel class map into polygons a frontend can
render and measure area from — the reason segmentation (not bounding boxes)
is the right tool for QuickQuote.

Saves to outputs/samples/: sample_XX.png (3-panel figure) and
polygons_XX.json (class, vertices, area in pixels).

Usage: python src/infer.py --n 8
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src.dataset import eval_transforms
from src.train import build_model, get_device

# background stays transparent; roof red-ish, vegetation green-ish
CLASS_COLORS = {1: (220, 60, 50), 2: (60, 170, 90)}


def mask_to_polygons(class_map: np.ndarray) -> list[dict]:
    polys = []
    for cls in (1, 2):
        binary = (class_map == cls).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < config.MIN_CONTOUR_AREA_PX:
                continue  # drop specks — tune MIN_CONTOUR_AREA_PX in config
            eps = config.POLY_EPSILON_FRAC * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, eps, True)
            polys.append({
                "class": config.CLASS_NAMES[cls],
                "area_px": float(area),
                "vertices": approx.reshape(-1, 2).tolist(),
            })
    return polys


def overlay(img: np.ndarray, class_map: np.ndarray, alpha=0.45) -> np.ndarray:
    out = img.copy()
    for cls, color in CLASS_COLORS.items():
        m = class_map == cls
        out[m] = ((1 - alpha) * out[m] + alpha * np.array(color)).astype(np.uint8)
    return out


def draw_polygons(img: np.ndarray, polys: list[dict]) -> np.ndarray:
    out = img.copy()
    for p in polys:
        cls = config.CLASS_NAMES.index(p["class"])
        pts = np.array(p["vertices"], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], True, CLASS_COLORS[cls], 2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()

    device = get_device()
    state = torch.load(config.CKPT_DIR / "best.pt", map_location=device)
    model = build_model().to(device)
    model.load_state_dict(state["model"])
    model.eval()
    tf = eval_transforms()

    # Pick tiles with the most roof content (QuickQuote's core class) plus a
    # couple random ones, so samples aren't cherry-picked to only easy cases.
    img_dir = config.TILES_DIR / "test" / "images"
    msk_dir = config.TILES_DIR / "test" / "masks"
    names = sorted(p.name for p in img_dir.glob("*.png"))
    roof_amount = []
    for name in names:
        m = cv2.imread(str(msk_dir / name), cv2.IMREAD_GRAYSCALE)
        roof_amount.append(((m == 1).mean(), name))
    roof_amount.sort(reverse=True)
    rng = np.random.default_rng(config.SEED)
    chosen = [n for _, n in roof_amount[: max(args.n - 2, 1)]]
    randoms = rng.choice(names, size=min(2, len(names)), replace=False)
    chosen += [n for n in randoms if n not in chosen]

    config.SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(chosen[: args.n]):
        bgr = cv2.imread(str(img_dir / name))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gt = cv2.imread(str(msk_dir / name), cv2.IMREAD_GRAYSCALE)
        x = tf(image=rgb, mask=gt)["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(x).argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
        polys = mask_to_polygons(pred)

        fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
        axes[0].imshow(rgb)
        axes[0].set_title("Input tile")
        axes[1].imshow(overlay(rgb, pred))
        axes[1].set_title("Predicted mask (roof=red, vegetation=green)")
        axes[2].imshow(draw_polygons(rgb, polys))
        axes[2].set_title(f"Extracted polygons ({len(polys)})")
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(config.SAMPLES_DIR / f"sample_{i:02d}.png", dpi=130)
        plt.close(fig)

        with open(config.SAMPLES_DIR / f"polygons_{i:02d}.json", "w") as f:
            json.dump({"tile": name, "polygons": polys}, f, indent=1)
        print(f"sample_{i:02d}: {name}  polygons={len(polys)}")

    print("Saved to", config.SAMPLES_DIR)


if __name__ == "__main__":
    main()
