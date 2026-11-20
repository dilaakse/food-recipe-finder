"""Query processing for the recipe search (course area B: classical NLP).

* text normalisation
* spelling correction with minimum edit distance against the recipe vocabulary
* regex extraction of constraints (time limit, diet)
* stemming (used by the TF-IDF baseline)
"""
import re
from collections import Counter

import numpy as np
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_stem = PorterStemmer().stem
STOP = set(ENGLISH_STOP_WORDS) | {"recipe", "recipes"}

# words that must never be "corrected"
KEEP = {"vegan", "vegetarian", "veggie", "quick", "easy", "under", "minutes", "fluffy", "juicy",
        "crispy", "homemade", "classic", "recipe", "recipes", "dinner", "sweet", "cold"}


def edit_distance(a, b):
    """Levenshtein distance (insert / delete / substitute, cost 1 each)."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def analyze(text):
    """Tokenizer for TF-IDF: lower-case words, stop words removed, Porter-stemmed."""
    return [_stem(w) for w in re.findall(r"[a-z]+", text.lower()) if w not in STOP]


class QueryProcessor:
    """Normalises queries using a vocabulary built from the recipe documents."""

    def __init__(self, docs):
        self.vocab = Counter(w for d in docs for w in re.findall(r"[a-z]+", d))

    def correct(self, token):
        """Replace an unknown word by the most frequent vocabulary word at edit distance 1."""
        if token in self.vocab or token in KEEP or len(token) < 4 or token.isdigit():
            return token
        cands = [w for w in self.vocab if abs(len(w) - len(token)) <= 1 and edit_distance(token, w) <= 1]
        return max(cands, key=self.vocab.get) if cands else token

    def normalize(self, query):
        query = re.sub(r"[^a-z0-9\s]", " ", query.lower())
        return " ".join(self.correct(t) for t in query.split())


def extract_constraints(query):
    """Regex extraction of a time limit in minutes ('max_min') and a diet ('diet')."""
    q, c = query.lower(), {}
    m = re.search(r"(?:under|less than|within|below)\s*(\d+)\s*(?:min|mins|minutes)", q)
    if m:
        c["max_min"] = int(m.group(1))
    elif re.search(r"\b(quick|quickly|fast|speedy)\b", q):
        c["max_min"] = 30
    if re.search(r"\bvegan\b", q):
        c["diet"] = "vegan"
    elif re.search(r"\b(vegetarian|veggie)\b", q):
        c["diet"] = "vegetarian"
    return c


def apply_filters(scores, constraints, recipes):
    """Push recipes that violate the extracted constraints below all others."""
    ok = np.ones(len(recipes), bool)
    if "max_min" in constraints:
        ok &= recipes["total_min"].values <= constraints["max_min"]
    if constraints.get("diet") == "vegan":
        ok &= recipes["is_vegan"].values
    if constraints.get("diet") == "vegetarian":
        ok &= recipes["is_vegetarian"].values
    s = scores.copy()
    s[~ok] -= 10
    return s
