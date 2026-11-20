"""Grad-CAM class-activation heatmaps (Selvaraju et al., 2017)."""
import numpy as np
import torch
import torch.nn.functional as F


def grad_cam(model, x, layer, class_idx=None):
    """Return (heatmap in [0, 1] with the input's height/width, class index, class probabilities).

    `model` must be a float (non-quantized) model: INT8 models cannot back-propagate.
    `layer` is the module whose output is used (for VGG16: `model.features[30]`).
    """
    model.eval()
    store = {}

    def forward_hook(module, inputs, output):
        store["a"] = output
        output.register_hook(lambda g: store.__setitem__("g", g))

    handle = layer.register_forward_hook(forward_hook)
    x = x.clone().requires_grad_(True)
    out = model(x)
    if class_idx is None:
        class_idx = out.argmax(1).item()
    model.zero_grad()
    out[0, class_idx].backward()
    handle.remove()

    weights = store["g"].mean(dim=(2, 3), keepdim=True)          # channel importance
    cam = torch.relu((weights * store["a"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0].detach()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam.cpu().numpy(), class_idx, torch.softmax(out, 1)[0].detach().cpu()


def overlay_heatmap(image, cam, alpha=0.45):
    """Blend a heatmap over an RGB image (float array in [0, 1], HxWx3). Returns uint8 HxWx3."""
    import matplotlib

    heat = matplotlib.colormaps["jet"](cam)[..., :3]
    return (((1 - alpha) * image + alpha * heat).clip(0, 1) * 255).astype(np.uint8)
