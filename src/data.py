"""Food-101 (10-class subset) dataset and data loaders."""
import json

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from . import config

CLS2IDX = {c: i for i, c in enumerate(config.CLASSES)}


def load_splits():
    with open(config.DATA_DIR / "splits.json") as f:
        return json.load(f)


class FoodDataset(Dataset):
    """ids look like 'pizza/12345' (Food-101 naming, without extension)."""

    def __init__(self, ids, transform):
        self.ids, self.transform = ids, transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        name = self.ids[i]
        img = Image.open(config.FOOD101_DIR / "images" / f"{name}.jpg").convert("RGB")
        return self.transform(img), CLS2IDX[name.split("/")[0]]


def train_transform(size):
    # data augmentation: random crop, horizontal flip, colour jitter
    return T.Compose([
        T.RandomResizedCrop(size, scale=(0.6, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.2, 0.2, 0.2),
        T.ToTensor(),
        T.Normalize(config.MEAN, config.STD),
    ])


def eval_transform(size):
    return T.Compose([
        T.Resize(int(size * 1.15)),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(config.MEAN, config.STD),
    ])


def make_loaders(size, batch=64, workers=2):
    """Return (train, val, test) loaders."""
    splits = load_splits()

    def mk(key, tf, shuffle):
        return DataLoader(FoodDataset(splits[key], tf), batch_size=batch, shuffle=shuffle,
                          num_workers=workers, pin_memory=True)

    return mk("train", train_transform(size), True), mk("val", eval_transform(size), False), \
        mk("test", eval_transform(size), False)
