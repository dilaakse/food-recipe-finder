"""Post-training static INT8 quantization of the image models (PyTorch FX graph mode)."""
import copy
from pathlib import Path

import torch
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx


def quantize_static(model, calib_loader, size, n_batches=8):
    """Quantize a float model to INT8. Activation ranges are calibrated on a few
    batches of the *validation* set (never the test set)."""
    engine = "x86" if "x86" in torch.backends.quantized.supported_engines else "fbgemm"
    torch.backends.quantized.engine = engine
    m = copy.deepcopy(model).cpu().eval()
    prepared = prepare_fx(m, get_default_qconfig_mapping(engine), (torch.randn(1, 3, size, size),))
    with torch.no_grad():
        for i, (x, _) in enumerate(calib_loader):
            prepared(x)
            if i + 1 >= n_batches:
                break
    return convert_fx(prepared)


def export_quantized(q_model, size, path):
    """Save a quantized model so the app can load it without recalibrating.

    Tries TorchScript first (`<path>.ts`); falls back to a pickled module (`<path>.pkl`).
    Returns the path that was written.
    """
    path = Path(path)
    example = torch.randn(1, 3, size, size)
    try:
        traced = torch.jit.trace(q_model, example, check_trace=False)
        with torch.no_grad():
            if not torch.allclose(traced(example), q_model(example), atol=1e-4):
                raise RuntimeError("traced model output differs from the original")
        out = path.with_suffix(".ts")
        traced.save(str(out))
        return out
    except Exception as e:  # noqa: BLE001
        print(f"TorchScript export failed ({e}); falling back to pickle")
        out = path.with_suffix(".pkl")
        torch.save(q_model, out)
        return out


def load_quantized(path):
    path = Path(path)
    if path.suffix == ".ts":
        return torch.jit.load(str(path), map_location="cpu").eval()
    return torch.load(path, map_location="cpu", weights_only=False).eval()
