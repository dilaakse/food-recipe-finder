# Recipes Through the Lens

*Food photo to recipe finder.* The app itself is called **Dish Detective**.

Upload a photo of a dish and describe what you want ("vegetarian, under 30 minutes"). The app
recognises the dish, shows where the model looked (Grad-CAM) and returns the top-5 matching recipes
with similarity scores. The photo model and the text model both run as **INT8-quantized** models.

## Quick start (run the app)

```bash
git clone https://github.com/dilaakse/food-recipe-finder
cd food-recipe-finder
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt   # tested: torch 2.9.1, torchvision 0.24.1, gradio 6.31.0, sentence-transformers 5.7.0
python app.py
```

Open the printed local URL. In Colab use `python app.py --share` to get a public link.
The first start downloads the model weights (see [Weights](#weights)) and the MiniLM embedding model
from the Hugging Face Hub. It runs on CPU; no GPU is needed.

## What is inside

| Part | Models | Course area |
|---|---|---|
| Dish recognition | Small CNN trained from scratch (baseline); VGG16 transfer learning (feature extraction, then fine-tuning); data augmentation + dropout; Grad-CAM | A – neural networks |
| Query processing | normalisation, spelling correction by minimum edit distance, regex extraction of time / diet constraints, stemming | B – classical NLP |
| Recipe search | TF-IDF + cosine (baseline) vs. MiniLM sentence embeddings (a 6-layer BERT-style Transformer, used as is, not fine-tuned) | B and C |
| Efficiency | static INT8 quantization of both CNNs (PyTorch FX), dynamic INT8 quantization of MiniLM, benchmark of size / latency / accuracy | D |

Data flow of the app:

```
photo ──► resize/normalise ──► VGG16 INT8 ──► dish + confidence ─────────┐
                  └────────────► VGG16 fp32 ──► Grad-CAM heatmap          │ (dish name is added to the query)
text ───► normalise + spelling correction ──► regex constraints ◄─────────┘
                  ├──► MiniLM INT8 embeddings ─┐
                  └──► TF-IDF (stemmed)        ├──► cosine similarity ──► top-5 recipes (+ optional diet/time filter)
```

The prediction is made by the INT8 VGG16. Grad-CAM needs gradients, which INT8 models cannot provide, so the
heatmap is computed on the float copy of the same network for the class predicted by the INT8 model.

## Data and sources

* **Images:** Food-101, 10-class subset (pizza, hamburger, pancakes, ice cream, french fries, omelette,
  chocolate cake, lasagna, tacos, caesar salad). 6,500 train / 1,000 validation (taken from the official train split)
  / 2,500 test images (official Food-101 test split). Bossard, Guillaumin, Van Gool, *Food-101 – Mining Discriminative
  Components with Random Forests*, ECCV 2014.
* **Recipes:** 342 recipes from the Food.com recipe data (Kaggle "Food.com – Recipes and Reviews"), read through the
  Hugging Face copy `GPTasty/PAID-recipes` (522,517 recipes; no licence is stated on that page). Recipes are assigned to the
  10 dishes by title and sampled per dish. The sample is deliberately enriched with vegetarian / vegan recipes and recipes
  of at most 30 minutes so that constrained queries have matches (`scripts/build_recipe_corpus.py`).
* **Test queries:** 30 hand-reviewed queries (`data/queries.csv`). The relevant recipes of each query (`data/qrels.csv`) are
  defined by rules on dish, diet tag, time limit and ingredient words (`scripts/build_queries.py`).
* **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` (Reimers & Gurevych, 2019; Wang et al., 2020).
* VGG16: Simonyan & Zisserman, 2015 (ImageNet weights from torchvision). Grad-CAM: Selvaraju et al., 2017.

## Evaluation protocol

The task has two inputs (a photo and a text request), so the models are compared in two groups. Inside each group
every model is evaluated on exactly the same held-out test set, with the same metrics and the same hardware.

| Step of the model ladder | Dish recognition (photo) | Recipe search (text) |
|---|---|---|
| 1. Classical baseline | – | TF-IDF + cosine (stemmed) |
| 2. Neural network trained by us | Small CNN from scratch; VGG16 fine-tuned (transfer learning) | – |
| 3. Transformer | – | MiniLM sentence embeddings (fp32) |
| 4. Quantized model | Small CNN INT8, VGG16 INT8 (static) | MiniLM INT8 (dynamic) |
| **Test set** | 2,500 official Food-101 test photos (10 classes), never used for training or model selection | 30 test queries with hand-reviewed relevant recipes (`data/qrels.csv`) over 342 recipes |
| **Metrics** | top-1 and top-5 accuracy | MRR, Hit@5, P@5 |
| **Efficiency** | model size (MB) and latency (ms), CPU, batch size 1, mean of 100 runs | model size (MB) and latency per query (ms), same CPU |

Model selection (number of epochs, fine-tuning) used the validation set only. Each quantized model is compared with its
own fp32 version on the same test set. The image models and the text models cannot share one test set because they take
different inputs, which is why the two groups have separate test sets.

## Results

All numbers were measured with the scripts in this repository (`results/*.json`). Latency: CPU, batch size 1, mean of
100 runs, Intel Xeon @ 2.00 GHz (Colab CPU runtime, 2 vCPUs). Size = size of the saved state dict.

**Dish recognition – 2,500 held-out test images**

| Model | Precision | Top-1 | Top-5 | Size (MB) | Latency (ms) |
|---|---|---|---|---|---|
| Small CNN (from scratch) | fp32 | 0.773 | 0.984 | 1.58 | 13.3 |
| Small CNN (from scratch) | int8 | 0.776 | 0.983 | 0.41 | 8.0 |
| VGG16 (fine-tuned) | fp32 | 0.904 | 0.994 | 110.27 | 432.1 |
| VGG16 (fine-tuned) | int8 | 0.904 | 0.994 | 27.68 | 281.3 |

Model selection used the validation set only: small CNN 0.709 (20 epochs) → 0.762 (40 epochs, used); VGG16 feature
extraction 0.848 → fine-tuning 0.886.

**Recipe search – 30 test queries over 342 recipes (text only, no metadata filter)**

| Model | Precision | MRR | Hit@5 | P@5 | Size (MB) | Latency per query (ms) |
|---|---|---|---|---|---|---|
| TF-IDF + cosine (stemmed) | – | 0.745 | 0.833 | 0.620 | 0.11 | 0.74 |
| MiniLM embeddings | fp32 | 0.952 | 0.967 | 0.833 | 90.89 | 18.27 |
| MiniLM embeddings | int8 | 0.936 | 0.967 | 0.787 | 58.63 | 14.24 |

Recall@5 is not used because it is capped when many recipes are relevant to a query (e.g. 36 for "pizza recipe").
Across two runs in different Colab sessions the TF-IDF and fp32 numbers were identical, the INT8 numbers differed slightly
(P@5 0.807 vs. 0.787) and latencies varied by 10–40 %; the tables above are from the last run.

## Reproduce everything

Training needs a GPU (Colab T4 is enough). Set `FRF_CKPT_DIR` (and optionally `FRF_FOOD101_DIR`, `FRF_WEIGHTS_DIR`)
to folders that survive a session restart, e.g. on Google Drive.

```bash
python -m scripts.prepare_food101          # download Food-101, keep 10 classes, create data/splits.json
python -m scripts.train_image_models       # small CNN (40 epochs) and VGG16 (5 + 8 epochs), checkpointed
python -m scripts.evaluate_image_models    # test accuracy, INT8 quantization, benchmark, exports weights/
python -m scripts.build_recipe_corpus      # data/recipes.csv from the Food.com data
python -m scripts.build_queries            # data/queries.csv and data/qrels.csv
python -m scripts.evaluate_retrieval       # results/retrieval_results.json and per-query table
```

`data/splits.json`, `data/recipes.csv`, `data/queries.csv` and `data/qrels.csv` are committed, so the evaluation scripts
use exactly the data of the reported results.

## Weights

Large files are not stored in git. `python app.py` downloads them automatically from Google Drive into `weights/`
(`vgg16_int8.ts`, about 28 MB, and `vgg16_fp32.pt`, about 110 MB). The Drive file ids are set in `src/config.py`
(`WEIGHT_FILES`). `scripts/evaluate_image_models.py` creates these files; to publish them, upload them to Google Drive,
share them with "Anyone with the link" and paste the ids into `WEIGHT_FILES`.

## Limitations

* The recipe relevance labels are rule-based and strict (e.g. only taco recipes count for "quick dinner with tortillas"),
  and the diet / time metadata is the same one used by the optional filter, so "with filter" results are optimistic.
* 30 queries and 342 recipes are small; differences of a few points (e.g. fp32 vs. INT8) are within noise.
* Dish labels of recipes come from their titles, so a few recipes are not typical examples of their dish.
* Softmax confidence of the image model is not calibrated: wrong predictions can have confidence close to 1.
* Dynamic quantization of MiniLM only converts linear layers; the word-embedding table stays in fp32, so the size
  reduction is limited (91 MB → 59 MB).
* Latency was measured on a 2-vCPU Colab CPU; other hardware will give different numbers.

## Repository layout

```
app.py                      Gradio app
src/                        data, models, training, quantization, benchmark, Grad-CAM, query processing, retrieval
scripts/                    reproducible pipeline (see "Reproduce everything")
data/                       split ids, recipe corpus, test queries and relevance labels
results/                    measured results (json / csv / figures)
```
