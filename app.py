"""Food photo -> recipe finder (Gradio app).

Run:  python app.py            (add --share for a public link, e.g. in Colab)

Pipeline
  photo  -> VGG16 (INT8) -> dish + confidence;  Grad-CAM heatmap from the float VGG16 copy
  text   -> normalise + spelling correction + regex constraints
         -> MiniLM sentence embeddings (INT8)  and  TF-IDF baseline  -> top-5 recipes each
"""
import argparse
import time

import gradio as gr
import numpy as np
import pandas as pd
import torch

from src import config
from src.data import eval_transform
from src.gradcam import grad_cam, overlay_heatmap
from src.models import make_vgg16
from src.quantize import load_quantized
from src.retrieval import DenseRetriever, TfidfRetriever, load_recipes
from src.text_pipeline import QueryProcessor, apply_filters, extract_constraints
from src.weights import ensure_weights

ensure_weights()

# ---- image models -------------------------------------------------------------------------
int8_path = next(p for p in (config.WEIGHTS_DIR / "vgg16_int8.ts", config.WEIGHTS_DIR / "vgg16_int8.pkl") if p.exists())
vgg_int8 = load_quantized(int8_path)                                   # used for the prediction
vgg_fp32 = make_vgg16(len(config.CLASSES), pretrained=False)           # used only for Grad-CAM
vgg_fp32.load_state_dict(torch.load(config.WEIGHTS_DIR / "vgg16_fp32.pt", map_location="cpu"))
vgg_fp32.eval()
IMG_TF = eval_transform(224)
MEAN, STD = torch.tensor(config.MEAN).view(3, 1, 1), torch.tensor(config.STD).view(3, 1, 1)

# ---- text models --------------------------------------------------------------------------
recipes = load_recipes()
qp = QueryProcessor(recipes["doc"])
tfidf = TfidfRetriever(recipes, qp)
dense_int8 = DenseRetriever(recipes, qp, quantize=True)

pretty = lambda c: c.replace("_", " ")


def search_table(retriever, query, constraints, use_filter, k=5):
    raw = retriever.scores(query)
    ranked = apply_filters(raw, constraints, recipes) if use_filter else raw
    order = np.argsort(-ranked)[:k]
    return pd.DataFrame({
        "recipe": recipes["title"].values[order],
        "dish": [pretty(d) for d in recipes["dish"].values[order]],
        "minutes": recipes["total_min"].values[order],
        "vegetarian": recipes["is_vegetarian"].values[order],
        "similarity": raw[order].round(3),
    })


def run(image, text, add_dish, use_filter):
    timing, label, heat, dish = [], None, None, ""
    if image is not None:
        x = IMG_TF(image.convert("RGB")).unsqueeze(0)
        t0 = time.perf_counter()
        with torch.no_grad():
            probs = torch.softmax(vgg_int8(x), 1)[0]
        timing.append(f"image classification (INT8 VGG16): {(time.perf_counter() - t0) * 1000:.0f} ms")
        pred = int(probs.argmax())
        label = {pretty(c): float(p) for c, p in zip(config.CLASSES, probs)}
        dish = pretty(config.CLASSES[pred]) if add_dish else ""
        t0 = time.perf_counter()
        cam, _, _ = grad_cam(vgg_fp32, x, vgg_fp32.features[30], class_idx=pred)
        shown = (x[0] * STD + MEAN).clamp(0, 1).permute(1, 2, 0).numpy()
        heat = overlay_heatmap(shown, cam)
        timing.append(f"Grad-CAM (float VGG16): {(time.perf_counter() - t0) * 1000:.0f} ms")

    query = f"{dish} {text or ''}".strip()
    if not query:
        return label, heat, "Upload a photo and/or type a request.", None, None, "  \n".join(timing)

    constraints = extract_constraints(query)
    info = (f"**Query used:** `{query}`  \n**After normalisation / spelling correction:** "
            f"`{qp.normalize(query)}`  \n**Extracted constraints:** `{constraints or 'none'}`"
            f"{'  (applied as filters)' if use_filter and constraints else ''}")
    tables = []
    for name, retriever in (("MiniLM INT8", dense_int8), ("TF-IDF", tfidf)):
        t0 = time.perf_counter()
        tables.append(search_table(retriever, query, constraints, use_filter))
        timing.append(f"retrieval ({name}): {(time.perf_counter() - t0) * 1000:.1f} ms")
    return label, heat, info, tables[0], tables[1], "  \n".join(timing)


with gr.Blocks(title="Food photo to recipe finder") as demo:
    gr.Markdown("# Food photo to recipe finder\nUpload a dish photo, describe what you want "
                "(e.g. *vegetarian, under 30 minutes*), and get the top-5 matching recipes.")
    with gr.Row():
        with gr.Column():
            img = gr.Image(type="pil", label="Food photo (optional)")
            txt = gr.Textbox(label="Your request", placeholder="vegetarian, under 30 minutes")
            add = gr.Checkbox(True, label="Add the recognised dish to the query")
            flt = gr.Checkbox(False, label="Apply extracted diet / time constraints as filters")
            btn = gr.Button("Find recipes", variant="primary")
        with gr.Column():
            label = gr.Label(num_top_classes=3, label="Recognised dish (VGG16, INT8)")
            heat = gr.Image(label="Grad-CAM: where the model looked (explains the INT8 prediction)")
    info = gr.Markdown()
    with gr.Row():
        t_dense = gr.Dataframe(label="Top-5 recipes: MiniLM sentence embeddings (INT8)")
        t_tfidf = gr.Dataframe(label="Top-5 recipes: TF-IDF baseline")
    timing = gr.Markdown()
    btn.click(run, [img, txt, add, flt], [label, heat, info, t_dense, t_tfidf, timing])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="create a public Gradio link")
    demo.launch(share=ap.parse_args().share)
