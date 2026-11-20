"""Train the image models.

Run:  python -m scripts.train_image_models            (both)
      python -m scripts.train_image_models --only small_cnn
      python -m scripts.train_image_models --only vgg16

Training is resumable: every epoch is checkpointed in CKPT_DIR.
"""
import argparse

import torch

from src import config
from src.data import make_loaders
from src.models import SmallCNN, make_vgg16
from src.train import fit


def train_small_cnn():
    """CNN from scratch, 128x128 input, 40 epochs (an earlier 20-epoch run reached 0.709 val accuracy)."""
    train_l, val_l, _ = make_loaders(128)
    _, hist = fit(SmallCNN(len(config.CLASSES)), "small_cnn_v2", train_l, val_l, epochs=40, lr=2e-3)
    print("best val accuracy:", max(h["val_acc"] for h in hist))


def train_vgg16():
    """Stage 1: feature extraction (new head only). Stage 2: fine-tune the last conv block."""
    train_l, val_l, _ = make_loaders(224, batch=32)
    vgg = make_vgg16(len(config.CLASSES), pretrained=True)
    vgg, h1 = fit(vgg, "vgg16_head", train_l, val_l, epochs=5, lr=1e-3)
    print("stage 1 best val accuracy:", max(h["val_acc"] for h in h1))

    vgg.load_state_dict(torch.load(config.CKPT_DIR / "vgg16_head_best.pt", map_location="cpu"))
    for p in vgg.features[24:].parameters():
        p.requires_grad = True
    vgg, h2 = fit(vgg, "vgg16_ft", train_l, val_l, epochs=8, lr=1e-4)
    print("stage 2 best val accuracy:", max(h["val_acc"] for h in h2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["small_cnn", "vgg16"])
    args = ap.parse_args()
    if args.only in (None, "small_cnn"):
        train_small_cnn()
    if args.only in (None, "vgg16"):
        train_vgg16()
