"""Evaluate the image models on the held-out test set, quantize them to INT8, benchmark
them and export the weights the app needs.

Run:  python -m scripts.evaluate_image_models

Accuracy of the float models is measured on the GPU if available (accuracy does not depend
on the device); INT8 models and all latencies are measured on the CPU, batch size 1,
averaged over 100 runs. Writes results/image_results.json and the files in WEIGHTS_DIR.
"""
import json

import pandas as pd
import torch

from src import config
from src.benchmark import cpu_name, image_latency_ms, size_mb, top1_top5
from src.data import make_loaders
from src.models import SmallCNN, make_vgg16
from src.quantize import export_quantized, load_quantized, quantize_static

SPECS = [
    # name, checkpoint, constructor, input size, exported file stem
    ("Small CNN v2", "small_cnn_v2_best.pt", lambda: SmallCNN(len(config.CLASSES)), 128, "small_cnn"),
    ("VGG16 fine-tuned", "vgg16_ft_best.pt", lambda: make_vgg16(len(config.CLASSES), pretrained=False), 224, "vgg16"),
]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config.WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, ckpt, ctor, size, stem in SPECS:
        model = ctor()
        model.load_state_dict(torch.load(config.CKPT_DIR / ckpt, map_location="cpu"))
        _, val_l, test_l = make_loaders(size, batch=32)

        t1, t5 = top1_top5(model, test_l, device)
        rows.append(dict(model=name, precision="fp32", top1=t1, top5=t5,
                         size_mb=size_mb(model.cpu()), latency_ms=image_latency_ms(model, size)))
        torch.save(model.cpu().state_dict(), config.WEIGHTS_DIR / f"{stem}_fp32.pt")

        q = quantize_static(model, val_l, size)          # calibration on validation data
        t1, t5 = top1_top5(q, test_l, "cpu")
        rows.append(dict(model=name, precision="int8", top1=t1, top5=t5,
                         size_mb=size_mb(q), latency_ms=image_latency_ms(q, size)))

        exported = export_quantized(q, size, config.WEIGHTS_DIR / f"{stem}_int8")
        reloaded = load_quantized(exported)             # sanity check: the exported file reproduces the model
        x = torch.randn(2, 3, size, size)
        print(f"{name}: exported {exported.name}, max abs diff after reload =",
              float((reloaded(x) - q(x)).abs().max()))
        print(rows[-2:])

    df = pd.DataFrame(rows)
    df.to_json(config.RESULTS_DIR / "image_results.json", orient="records", indent=2)
    print("CPU:", cpu_name())
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
