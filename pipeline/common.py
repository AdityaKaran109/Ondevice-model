"""Shared constants and helpers for the FFNet-40S on-device pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
ANDROID_ASSETS_DIR = ROOT / "android" / "app" / "src" / "main" / "assets"
AIMET_CONFIG = ROOT / "Notebook" / "ffnet_aimet_config.json"

# FFNet-40S native resolution (Cityscapes). Output is 1/8 of this: 128 x 256.
HEIGHT, WIDTH = 1024, 2048
INPUT_SHAPE = (1, 3, HEIGHT, WIDTH)

TARGET_DEVICE = "Samsung Galaxy S25 Ultra"
TFLITE_NAME = "ffnet_40s_w8a8.tflite"
# Same network compiled for upright (portrait) frames: 1024 wide x 2048 tall.
PORTRAIT_TFLITE_NAME = "ffnet_40s_w8a8_portrait.tflite"

CITYSCAPES_CLASSES = [
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
    "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
    "truck", "bus", "train", "motorcycle", "bicycle",
]

CITYSCAPES_PALETTE = [
    (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156),
    (190, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0),
    (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
    (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100),
    (0, 0, 230), (119, 11, 32),
]

_to_tensor = transforms.ToTensor()


def load_urbansyn(num_test: int = 10, seed: int = 0):
    """Load ~100 UrbanSyn RGB frames and split into (calibration, test) sets."""
    from datasets import load_dataset

    ds = load_dataset("UrbanSyn/UrbanSyn", split="train", data_files="rgb/*_00*.png")
    split = ds.train_test_split(test_size=num_test, seed=seed)
    return split["train"], split["test"]


def fit_to(image: Image.Image, size: tuple[int, int] = (WIDTH, HEIGHT)) -> Image.Image:
    """Center-crop `image` to the aspect ratio of `size` (width, height), then resize to it."""
    image = image.convert("RGB")
    w, h = image.size
    target_w, target_h = size
    if w * target_h > h * target_w:  # too wide
        crop_w = h * target_w // target_h
        image = image.crop(((w - crop_w) // 2, 0, (w + crop_w) // 2, h))
    elif w * target_h < h * target_w:  # too tall
        crop_h = w * target_h // target_w
        image = image.crop((0, (h - crop_h) // 2, w, (h + crop_h) // 2))
    return image if image.size == size else image.resize(size, Image.BILINEAR)


def preprocess(image: Image.Image, size: tuple[int, int] = (WIDTH, HEIGHT)) -> torch.Tensor:
    """PIL RGB image -> [1, 3, H, W] float tensor in [0, 1] at model resolution.

    The FFNet40S wrapper applies ImageNet normalization inside forward(),
    so the phone only has to scale pixels to [0, 1] too.
    """
    return _to_tensor(fit_to(image, size)).unsqueeze(0)


def logits_to_mask(logits: torch.Tensor | np.ndarray, channel_last: bool = False) -> np.ndarray:
    """Logits [1, C, h, w] (or [1, h, w, C]) -> class-id mask [h, w] uint8."""
    arr = logits.detach().cpu().numpy() if isinstance(logits, torch.Tensor) else np.asarray(logits)
    axis = -1 if channel_last else 1
    return arr.argmax(axis=axis)[0].astype(np.uint8)


def colorize(mask: np.ndarray) -> Image.Image:
    palette = np.zeros((256, 3), dtype=np.uint8)
    palette[: len(CITYSCAPES_PALETTE)] = CITYSCAPES_PALETTE
    return Image.fromarray(palette[mask])


def overlay(image: Image.Image, mask: np.ndarray, alpha: float = 0.5) -> Image.Image:
    base = fit_to(image, (mask.shape[1] * 8, mask.shape[0] * 8))
    color = colorize(mask).resize(base.size, Image.NEAREST)
    return Image.blend(base, color, alpha)


def mean_iou(pred: np.ndarray, ref: np.ndarray, num_classes: int = 19) -> float:
    """mIoU of `pred` against `ref`, over classes present in either."""
    ious = []
    for c in range(num_classes):
        p, r = pred == c, ref == c
        union = np.logical_or(p, r).sum()
        if union:
            ious.append(np.logical_and(p, r).sum() / union)
    return float(np.mean(ious)) if ious else float("nan")
