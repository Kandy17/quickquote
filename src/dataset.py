"""PyTorch Dataset + albumentations pipelines for the prepared tiles."""
import sys
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


def train_transforms() -> A.Compose:
    # Aerial imagery has no canonical "up", so flips/rot90 are label-safe and
    # effectively grow the dataset. Brightness/contrast jitter simulates
    # different capture conditions.
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2,
                                   contrast_limit=0.2, p=0.5),
        A.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ToTensorV2(),
    ])


def eval_transforms() -> A.Compose:
    # No augmentation at eval time — we measure the model, not the jitter.
    return A.Compose([
        A.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ToTensorV2(),
    ])


class TileDataset(Dataset):
    def __init__(self, split: str, transforms: A.Compose):
        self.img_dir = config.TILES_DIR / split / "images"
        self.msk_dir = config.TILES_DIR / split / "masks"
        self.names = sorted(p.name for p in self.img_dir.glob("*.png"))
        if not self.names:
            raise FileNotFoundError(
                f"No tiles in {self.img_dir}. Run src/prepare_data.py first.")
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, i: int):
        name = self.names[i]
        img = cv2.cvtColor(cv2.imread(str(self.img_dir / name)),
                           cv2.COLOR_BGR2RGB)
        msk = cv2.imread(str(self.msk_dir / name), cv2.IMREAD_GRAYSCALE)
        out = self.transforms(image=img, mask=msk)
        return out["image"], out["mask"].long(), name
