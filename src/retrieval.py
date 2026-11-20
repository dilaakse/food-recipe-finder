"""Recipe retrieval models (TF-IDF baseline, MiniLM embeddings fp32 / INT8) and evaluation."""
import pickle

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from . import config
from .text_pipeline import QueryProcessor, analyze


def load_recipes(path=None):
    """Load the recipe corpus and build the text every model sees for a recipe
    (title, category, keywords, ingredients; instructions are left out on purpose)."""
    recipes = pd.read_csv(path or config.DATA_DIR / "recipes.csv")
    for c in ["title", "category", "keywords", "ingredients"]:
        recipes[c] = recipes[c].fillna("")
    recipes["doc"] = (recipes["title"] + ". " + recipes["category"] + ". " +
                      recipes["keywords"] + ". " + recipes["ingredients"]).str.lower()
    return recipes


class TfidfRetriever:
    """Classical baseline: stemmed TF-IDF vectors compared with cosine similarity."""

    name = "TF-IDF + cosine"

    def __init__(self, recipes, qp: QueryProcessor):
        self.qp = qp
        self.vec = TfidfVectorizer(analyzer=analyze, sublinear_tf=True)
        self.X = self.vec.fit_transform(recipes["doc"])  # L2-normalised rows -> dot product = cosine

    def scores(self, query):
        return (self.X @ self.vec.transform([self.qp.normalize(query)]).T).toarray().ravel()

    def size_mb(self):
        return len(pickle.dumps((self.vec, self.X))) / 1e6


class DenseRetriever:
    """Sentence-embedding retrieval with a BERT-style model (MiniLM). With quantize=True the
    linear layers are dynamically quantized to INT8; documents and queries are then both
    encoded by the quantized model."""

    def __init__(self, recipes, qp: QueryProcessor, quantize=False, model_name=config.EMBED_MODEL):
        import copy

        import torch
        from sentence_transformers import SentenceTransformer

        self.qp = qp
        self.name = "MiniLM " + ("int8" if quantize else "fp32")
        model = SentenceTransformer(model_name, device="cpu")
        if quantize:
            model = torch.ao.quantization.quantize_dynamic(copy.deepcopy(model), {torch.nn.Linear},
                                                           dtype=torch.qint8)
        self.model = model
        self.E = model.encode(recipes["doc"].tolist(), batch_size=32, normalize_embeddings=True,
                              show_progress_bar=False)

    def scores(self, query):
        v = self.model.encode([self.qp.normalize(query)], normalize_embeddings=True,
                              show_progress_bar=False)[0]
        return self.E @ v

    def size_mb(self):
        import os
        import tempfile

        import torch

        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            torch.save(self.model.state_dict(), f.name)
            return os.path.getsize(f.name) / 1e6


def top_k(scores, recipes, k=5):
    order = np.argsort(-scores)[:k]
    return recipes.iloc[order].assign(score=scores[order])


def load_eval_data():
    queries = pd.read_csv(config.DATA_DIR / "queries.csv")
    qrels = pd.read_csv(config.DATA_DIR / "qrels.csv")
    rel = qrels.groupby("query_id")["recipe_id"].apply(set).to_dict()
    return queries, rel


def evaluate(score_fn, recipes, queries, rel, k=5):
    """MRR, Hit@k and Precision@k over all queries."""
    rr, hit, prec = [], [], []
    ids_all = recipes["recipe_id"].values
    for qid, text in zip(queries["query_id"], queries["query"]):
        ids = ids_all[np.argsort(-score_fn(text))]
        ranks = [i for i, r in enumerate(ids, 1) if r in rel[qid]]
        rr.append(1 / ranks[0])
        hit.append(float(ranks[0] <= k))
        prec.append(sum(r in rel[qid] for r in ids[:k]) / k)
    return dict(MRR=round(float(np.mean(rr)), 4), Hit_at_5=round(float(np.mean(hit)), 4),
                P_at_5=round(float(np.mean(prec)), 4))


def per_query(score_fn, recipes, queries, rel):
    """For every query: rank of the first relevant recipe and the title ranked first."""
    ids_all = recipes["recipe_id"].values
    out = []
    for qid, text in zip(queries["query_id"], queries["query"]):
        order = np.argsort(-score_fn(text))
        ids = ids_all[order]
        rank = next(i for i, r in enumerate(ids, 1) if r in rel[qid])
        out.append((rank, recipes["title"].values[order[0]]))
    return out
