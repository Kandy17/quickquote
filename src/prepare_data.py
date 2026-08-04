"""Tile the LandCover.ai orthophotos into 256x256 training tiles.

- Remaps labels to 3 classes: 0=background, 1=roof, 2=vegetation.
- Splits 70/15/15 at the SOURCE IMAGE level to avoid spatial leakage.
- Filters mostly-empty tiles, caps split sizes (see config).

Usage: python src/prepare_data.py
Output: data/tiles/{train,val,test}/{images,masks}/*.png + tile_manifest.json
"""
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


def remap_mask(mask: np.ndarray) -> np.ndarray:
    """LandCover.ai: 0=bg, 1=building, 2=woodland, 3=water(, 4=road).
    QuickQuote v1: 0=background, 1=roof, 2=vegetation. Water/roads -> background.
    """
    out = np.zeros_like(mask, dtype=np.uint8)
    out[mask == 1] = 1
    out[mask == 2] = 2
    return out


def split_images(stems: list[str]) -> dict[str, list[str]]:
    rng = random.Random(config.SEED)
    stems = sorted(stems)
    rng.shuffle(stems)
    n = len(stems)
    n_train = round(n * config.SPLIT_FRACS[0])
    n_val = round(n * config.SPLIT_FRACS[1])
    return {
        "train": stems[:n_train],
        "val": stems[n_train:n_train + n_val],
        "test": stems[n_train + n_val:],
    }


def tile_one(stem: str, split: str, rng: random.Random, manifest: list) -> int:
    img_path = config.RAW_DIR / "images" / f"{stem}.tif"
    msk_path = config.RAW_DIR / "masks" / f"{stem}.tif"
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    msk = cv2.imread(str(msk_path), cv2.IMREAD_GRAYSCALE)
    if img is None or msk is None:
        print(f"WARNING: could not read {stem}, skipping")
        return 0
    msk = remap_mask(msk)
    ts, stride = config.TILE_SIZE, config.TILE_STRIDE
    h, w = msk.shape
    kept = 0
    for y in range(0, h - ts + 1, stride):
        for x in range(0, w - ts + 1, stride):
            m = msk[y:y + ts, x:x + ts]
            roof_frac = float((m == 1).mean())
            veg_frac = float((m == 2).mean())
            keep = (
                roof_frac >= config.MIN_ROOF_FRAC
                or veg_frac >= config.MIN_VEG_FRAC
                or rng.random() < config.BG_KEEP_PROB
            )
            if not keep:
                continue
            name = f"{stem}_{y}_{x}.png"
            cv2.imwrite(str(config.TILES_DIR / split / "images" / name),
                        img[y:y + ts, x:x + ts])
            cv2.imwrite(str(config.TILES_DIR / split / "masks" / name), m)
            manifest.append({"tile": name, "split": split, "source": stem,
                             "roof_frac": round(roof_frac, 4),
                             "veg_frac": round(veg_frac, 4)})
            kept += 1
    return kept


def cap_split(split: str, manifest: list) -> list:
    """Deterministically subsample a split to its cap, preferring roof tiles
    (roofs are the rarest class and the one QuickQuote cares most about)."""
    cap = config.MAX_TILES[split]
    entries = [e for e in manifest if e["split"] == split]
    if len(entries) <= cap:
        return manifest
    rng = random.Random(config.SEED + hash(split) % 1000)
    roof = [e for e in entries if e["roof_frac"] >= config.MIN_ROOF_FRAC]
    rest = [e for e in entries if e["roof_frac"] < config.MIN_ROOF_FRAC]
    rng.shuffle(roof)
    rng.shuffle(rest)
    keep = (roof + rest)[:cap]
    keep_names = {e["tile"] for e in keep}
    for e in entries:
        if e["tile"] not in keep_names:
            (config.TILES_DIR / split / "images" / e["tile"]).unlink(missing_ok=True)
            (config.TILES_DIR / split / "masks" / e["tile"]).unlink(missing_ok=True)
    return [e for e in manifest if e["split"] != split] + keep


def main() -> None:
    stems = [p.stem for p in (config.RAW_DIR / "images").glob("*.tif")]
    if not stems:
        sys.exit("No raw images found. Run src/download_data.py first.")
    splits = split_images(stems)
    print({k: len(v) for k, v in splits.items()}, "source images per split")

    for split in ("train", "val", "test"):
        for sub in ("images", "masks"):
            (config.TILES_DIR / split / sub).mkdir(parents=True, exist_ok=True)

    rng = random.Random(config.SEED)
    manifest: list = []
    for split, split_stems in splits.items():
        for stem in split_stems:
            n = tile_one(stem, split, rng, manifest)
            print(f"{split}: {stem} -> {n} tiles kept")

    for split in ("train", "val", "test"):
        manifest = cap_split(split, manifest)

    counts = {s: sum(1 for e in manifest if e["split"] == s)
              for s in ("train", "val", "test")}
    meta = {"counts": counts, "source_split": splits,
            "class_names": config.CLASS_NAMES, "tile_size": config.TILE_SIZE,
            "seed": config.SEED}
    with open(config.TILES_DIR / "tile_manifest.json", "w") as f:
        json.dump({"meta": meta, "tiles": manifest}, f, indent=1)
    print("Final tile counts:", counts)


if __name__ == "__main__":
    main()
