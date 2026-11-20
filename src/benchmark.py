"""Accuracy, model size and latency measurements."""
import os
import platform
import tempfile
import time

import numpy as np
import torch


def cpu_name():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":")[1].strip()
    except OSError:
        pass
    return platform.processor()


def size_mb(model):
    """Size of the saved state dict in MB (same method for every model)."""
    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        torch.save(model.state_dict(), f.name)
        return os.path.getsize(f.name) / 1e6


@torch.no_grad()
def top1_top5(model, loader, device):
    model.eval().to(device)
    c1 = c5 = n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        top5 = model(x).topk(5, 1).indices
        c1 += (top5[:, 0] == y).sum().item()
        c5 += (top5 == y[:, None]).any(1).sum().item()
        n += len(y)
    return c1 / n, c5 / n


@torch.no_grad()
def image_latency_ms(model, size, runs=100, warmup=10):
    """Mean CPU latency for one image (batch size 1) over `runs` runs."""
    model.eval().cpu()
    x = torch.randn(1, 3, size, size)
    for _ in range(warmup):
        model(x)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model(x)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times))


def query_latency_ms(score_fn, query="vegetarian pizza under 30 minutes", runs=100, warmup=10):
    """Mean latency of scoring one text query (normalisation + encoding + ranking)."""
    for _ in range(warmup):
        score_fn(query)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        score_fn(query)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times))
