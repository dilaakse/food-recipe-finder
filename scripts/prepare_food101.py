"""Download Food-101, keep the 10 selected classes and create the train/val/test split.

Run:  python -m scripts.prepare_food101

The split (data/splits.json) is committed to the repository. If it already exists
it is kept as is, so that every run uses the same train/val/test images.
"""
import json
import random
import subprocess
import urllib.request

from src import config

URL = "https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
VAL_PER_CLASS = 100


def main():
    work = config.FOOD101_DIR.parent
    work.mkdir(parents=True, exist_ok=True)

    if not config.FOOD101_DIR.exists():
        archive = work / "food-101.tar.gz"
        if not archive.exists():
            print("downloading Food-101 (about 5 GB) ...")
            urllib.request.urlretrieve(URL, archive)
        members = ["food-101/meta"] + [f"food-101/images/{c}" for c in config.CLASSES]
        subprocess.run(["tar", "-xzf", str(archive), "-C", str(work)] + members, check=True)
        archive.unlink()

    splits_path = config.DATA_DIR / "splits.json"
    if splits_path.exists():
        print("data/splits.json already exists, keeping it")
        return

    def read(name):
        with open(config.FOOD101_DIR / "meta" / name) as f:
            return [line.strip() for line in f]

    train_all = [x for x in read("train.txt") if x.split("/")[0] in config.CLASSES]
    test = [x for x in read("test.txt") if x.split("/")[0] in config.CLASSES]   # official Food-101 test split

    random.seed(config.SEED)
    train, val = [], []
    for c in config.CLASSES:
        items = [x for x in train_all if x.startswith(c + "/")]
        random.shuffle(items)
        val += items[:VAL_PER_CLASS]
        train += items[VAL_PER_CLASS:]

    config.DATA_DIR.mkdir(exist_ok=True)
    with open(splits_path, "w") as f:
        json.dump({"classes": config.CLASSES, "train": train, "val": val, "test": test}, f)
    print(f"train {len(train)}  val {len(val)}  test {len(test)}")


if __name__ == "__main__":
    main()
