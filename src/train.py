"""Training loop with per-epoch checkpoints (resumable after a Colab disconnect)."""
import json
import time

import torch
import torch.nn as nn

from . import config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def evaluate(model, loader):
    """Return (loss, accuracy) on a loader."""
    model.eval()
    crit = nn.CrossEntropyLoss(reduction="sum")
    loss_sum, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss_sum += crit(out, y).item()
            correct += (out.argmax(1) == y).sum().item()
            n += len(y)
    return loss_sum / n, correct / n


def fit(model, name, train_loader, val_loader, epochs, lr, weight_decay=1e-4):
    """Train with AdamW + cosine schedule + mixed precision.

    Saves `<name>_last.pt` (full state, used to resume) and `<name>_best.pt`
    (weights with the best validation accuracy) in CKPT_DIR. Returns the model
    loaded with the best weights and the history list.
    """
    config.CKPT_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    last, best = config.CKPT_DIR / f"{name}_last.pt", config.CKPT_DIR / f"{name}_best.pt"

    model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))
    crit = nn.CrossEntropyLoss()

    start, best_acc, hist = 0, 0.0, []
    if last.exists():
        s = torch.load(last, map_location=DEVICE)
        model.load_state_dict(s["model"])
        opt.load_state_dict(s["opt"])
        sched.load_state_dict(s["sched"])
        start, best_acc, hist = s["epoch"] + 1, s["best_acc"], s["hist"]
        print(f"checkpoint found, resuming from epoch {start}")

    for ep in range(start, epochs):
        model.train()
        t0, tl, tc, n = time.time(), 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEVICE == "cuda")):
                out = model(x)
                loss = crit(out, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tl += loss.item() * len(y)
            tc += (out.argmax(1) == y).sum().item()
            n += len(y)
        sched.step()
        val_loss, val_acc = evaluate(model, val_loader)
        hist.append(dict(epoch=ep, train_loss=tl / n, train_acc=tc / n, val_loss=val_loss, val_acc=val_acc))
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best)
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                        epoch=ep, best_acc=best_acc, hist=hist), last)
        print(f"epoch {ep + 1}/{epochs}  train_acc {tc / n:.3f}  val_acc {val_acc:.3f}  ({time.time() - t0:.0f} s)")

    with open(config.RESULTS_DIR / f"{name}_history.json", "w") as f:
        json.dump(hist, f)
    model.load_state_dict(torch.load(best, map_location=DEVICE))
    return model, hist
