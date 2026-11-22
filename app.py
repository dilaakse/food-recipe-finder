"""Dish Detective - food photo -> recipe finder (Gradio app).

Run:  python app.py            (add --share for a public link, e.g. in Colab)

Pipeline
  photo  -> VGG16 (INT8) -> dish + confidence;  Grad-CAM heatmap from the float VGG16 copy
  text   -> normalise + spelling correction + regex constraints
         -> MiniLM sentence embeddings (INT8)  and  TF-IDF baseline  -> top-5 recipes each
"""
import argparse
import html
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


def search(retriever, query, constraints, use_filter, k=5):
    """Return (row positions, raw similarity scores) of the top-k recipes."""
    raw = retriever.scores(query)
    ranked = apply_filters(raw, constraints, recipes) if use_filter else raw
    order = np.argsort(-ranked)[:k]
    if use_filter and constraints:
        order = order[ranked[order] > -5]       # drop recipes that break the time / diet limit
    return order, raw


def search_table(order, raw):
    return pd.DataFrame({
        "similarity": raw[order].astype(float).round(3),
        "recipe": recipes["title"].values[order],
        "min": recipes["total_min"].values[order],
        "veg": recipes["is_vegetarian"].values[order],
        "dish": [pretty(d) for d in recipes["dish"].values[order]],
    })


def recipe_cards(order, raw):
    """Recipe results as expandable cards: click a card to read ingredients and steps."""
    if len(order) == 0:
        return "<p>No recipes found.</p>"
    cards = []
    for i in order:
        r = recipes.iloc[i]
        badges = f"⏱ {int(r['total_min'])} min" + ("  ·  🌱 vegetarian" if r["is_vegetarian"] else "")
        ingredients = "".join(f"<li>{html.escape(x.strip())}</li>" for x in str(r["ingredients"]).split(";") if x.strip())
        cards.append(
            "<details style='border:1px solid var(--border-color-primary);border-radius:10px;"
            "padding:10px 14px;margin:8px 0;background:var(--background-fill-secondary)'>"
            "<summary style='cursor:pointer;font-weight:600;font-size:1.05em'>"
            f"{html.escape(str(r['title']))}"
            f"<span style='font-weight:400;opacity:.7;font-size:.85em;margin-left:10px'>{badges}"
            f"  ·  match {raw[i]:.2f}</span></summary>"
            f"<div style='margin-top:10px'><b>Ingredients</b><ul>{ingredients}</ul>"
            f"<b>Directions</b><p>{html.escape(str(r['instructions']))}</p></div></details>")
    return "".join(cards)


def run(image, text, add_dish, use_filter):
    timing, label, heat, dish = [], None, None, ""
    if image is not None:
        x = IMG_TF(image.convert("RGB")).unsqueeze(0)
        t0 = time.perf_counter()
        with torch.no_grad():
            probs = torch.softmax(vgg_int8(x), 1)[0]
        timing.append(f"image classification (INT8 VGG16): {(time.perf_counter() - t0) * 1000:.0f} ms")
        pred = int(probs.argmax())
        top = sorted(zip(config.CLASSES, probs.tolist()), key=lambda t: -t[1])[:3]
        label = {pretty(c): p for c, p in top if p >= 0.01} or {pretty(top[0][0]): top[0][1]}   # hide near-0 % classes
        dish = pretty(config.CLASSES[pred]) if add_dish else ""
        t0 = time.perf_counter()
        cam, _, _ = grad_cam(vgg_fp32, x, vgg_fp32.features[30], class_idx=pred)
        shown = (x[0] * STD + MEAN).clamp(0, 1).permute(1, 2, 0).numpy()
        heat = overlay_heatmap(shown, cam)
        timing.append(f"Grad-CAM (float VGG16): {(time.perf_counter() - t0) * 1000:.0f} ms")

    if dish and dish in (text or "").lower():       # the user already named the dish
        dish = ""
    query = f"{dish} {text or ''}".strip()
    if not query:
        return label, heat, "Upload a photo and/or type a request.", "", None, "  \n".join(timing)

    constraints = extract_constraints(qp.normalize(query))   # extracted after spelling correction
    info = (f"**Search text:** `{query}`  \n**After spelling correction:** `{qp.normalize(query)}`  \n"
            f"**Detected preferences:** `{constraints or 'none'}`"
            f"{'  (used as filters)' if use_filter and constraints else ''}")
    results = []
    for name, retriever in (("MiniLM INT8", dense_int8), ("TF-IDF", tfidf)):
        t0 = time.perf_counter()
        results.append(search(retriever, query, constraints, use_filter))
        timing.append(f"retrieval ({name}): {(time.perf_counter() - t0) * 1000:.1f} ms")
    return (label, heat, info, recipe_cards(*results[0]), search_table(*results[1]), "  \n".join(timing))


with gr.Blocks(title="Dish Detective") as demo:
    gr.Markdown("# 🕵️ Dish Detective\nTake a photo of a dish, tell us what you feel like cooking "
                "(for example *vegetarian, under 30 minutes*) and get recipe ideas.")
    with gr.Row():
        with gr.Column():
            img = gr.Image(type="pil", label="Your dish photo (optional)")
            txt = gr.Textbox(label="What are you in the mood for?", placeholder="e.g. vegetarian, under 30 minutes")
            add = gr.Checkbox(True, label="Use the dish in my photo as part of the search")
            flt = gr.Checkbox(True, label="Only show recipes that match my diet and time limit")
            btn = gr.Button("Find recipes", variant="primary")
        with gr.Column():
            label = gr.Label(num_top_classes=3, label="What we see in your photo (VGG16, INT8)")
            heat = gr.Image(label="What the model looked at (Grad-CAM)")
    gr.Markdown("### Recipe ideas\n*Best matches from MiniLM sentence embeddings (INT8). Click a recipe to open it.*")
    cards = gr.HTML()
    with gr.Accordion("Behind the scenes: compare with a plain keyword search", open=True):
        info = gr.Markdown()
        t_tfidf = gr.Dataframe(label="Keyword search (TF-IDF baseline)", wrap=True)
        timing = gr.Markdown()
    btn.click(run, [img, txt, add, flt], [label, heat, info, cards, t_tfidf, timing])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="create a public Gradio link")
    demo.launch(share=ap.parse_args().share)
