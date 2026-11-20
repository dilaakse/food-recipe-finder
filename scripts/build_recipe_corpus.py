"""Build the recipe corpus (data/recipes.csv) from the public Food.com recipe data.

Run:  python -m scripts.build_recipe_corpus

Source: Food.com recipes (Kaggle "Food.com - Recipes and Reviews"), read through the
Hugging Face copy `GPTasty/PAID-recipes` (522,517 recipes). Recipes are assigned to one of
the 10 dishes by their title, filtered for quality, and sampled per dish. The sample is
deliberately enriched with vegetarian / vegan recipes and recipes that take at most
30 minutes, so that queries with diet or time constraints have matching recipes.
"""
import re

import numpy as np
import pandas as pd

from src import config

SOURCE = "GPTasty/PAID-recipes"
USECOLS = ["RecipeId", "Name", "CookTime", "PrepTime", "TotalTime", "Description", "RecipeCategory",
           "Keywords", "RecipeIngredientParts", "RecipeInstructions", "AggregatedRating", "ReviewCount"]

# dish -> (include regex, exclude regex), matched against the lower-cased recipe title
DISHES = {
    "pizza": (r"\bpizza\b", r"sauce|dough|crust|seasoning|burger|casserole|pasta|bagel|bites|rolls?\b|dip|pockets|lasagna|salad|soup|cups|muffin"),
    "hamburger": (r"\b(hamburger|burger)s?\b", r"sauce|seasoning|casserole|soup|pizza|helper|dip|pie|bake|hash|stroganoff"),
    "pancakes": (r"\bpancakes?\b", r"syrup|mix\b|waffle|sauce|bake|casserole|roll"),
    "ice_cream": (r"\bice cream\b", r"cake|pie|sandwich|cupcake|sauce|topping|casserole|float|bread|soda|cone|sundae|bars?\b"),
    "french_fries": (r"\bfries\b", r"stir|rice|chicken|dip|sauce|seasoning|poutine|casserole|burger|chili|fish|nachos"),
    "omelette": (r"\bomel(et|ette)s?\b", r"muffin|casserole|sandwich|pie|cups|roll|cake"),
    "chocolate_cake": (r"\bchocolate cake\b", r"frosting|icing|glaze|sauce|cheesecake|cupcakes?|trifle|balls|pops|truffle|ice cream|pie|poke|mix\b|filling"),
    "lasagna": (r"\blasagn[ae]\b", r"soup|rolls?\b|roll-ups|dip|sauce|salad|cups|bites|pizza|skillet|stuffed|cupcakes|noodles?\b"),
    "tacos": (r"\btacos?\b", r"salad|soup|pizza|casserole|dip|seasoning|sauce|bake|bites|lasagna|cups|pasta|spaghetti|bowl|rolls?\b|cupcakes|meat\b|shells?\b"),
    "caesar_salad": (r"\bcaesar\b.*\bsalad\b|\bsalad\b.*\bcaesar\b", r"dressing|pasta|pizza|wrap|sandwich|dip|soup|croutons"),
}
PATTERNS = {d: (re.compile(i), re.compile(x)) for d, (i, x) in DISHES.items()}


def match_dish(name):
    n = str(name).lower()
    hits = [d for d, (inc, exc) in PATTERNS.items() if inc.search(n) and not exc.search(n)]
    return hits[0] if len(hits) == 1 else None


def to_list(x):
    """Handle parquet lists/arrays and R-style c("a", "b") strings."""
    if x is None:
        return []
    if isinstance(x, (list, tuple, np.ndarray)):
        return [str(i) for i in x]
    if isinstance(x, str):
        if not x.strip() or x == "character(0)":
            return []
        if x.startswith("c("):
            return [y.replace('\\"', '"') for y in re.findall(r'"((?:[^"\\]|\\.)*)"', x)]
        return [x]
    return []


def to_minutes(s):
    """ISO-8601 duration like PT1H30M -> minutes; plain numbers are treated as minutes."""
    if isinstance(s, (int, float, np.integer, np.floating)):
        return float(s)
    if not isinstance(s, str):
        return np.nan
    m = re.fullmatch(r"P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?", s)
    if not m:
        return np.nan
    d, h, mi = (int(g) if g else 0 for g in m.groups())
    return d * 1440 + h * 60 + mi


def main():
    from datasets import load_dataset

    ds = load_dataset(SOURCE, split="train")
    print("recipes in source:", ds.num_rows)

    idx, dishes = [], []
    for i, n in enumerate(ds["Name"]):
        d = match_dish(n)
        if d:
            idx.append(i)
            dishes.append(d)
    cand = ds.select(idx).select_columns(USECOLS).to_pandas()
    cand["dish"] = dishes
    print("title matches per dish:\n", cand["dish"].value_counts())

    cand["keywords"] = cand["Keywords"].map(to_list)
    cand["ingredients"] = cand["RecipeIngredientParts"].map(to_list)
    cand["steps"] = cand["RecipeInstructions"].map(to_list)
    cand["total_min"] = cand["TotalTime"].map(to_minutes)
    kw_lower = cand["keywords"].map(lambda ks: {k.lower() for k in ks})
    name_lower = cand["Name"].str.lower()
    cand["is_vegan"] = kw_lower.map(lambda s: "vegan" in s) | name_lower.str.contains("vegan")
    cand["is_vegetarian"] = (cand["is_vegan"] | kw_lower.map(lambda s: "vegetarian" in s)
                             | name_lower.str.contains("vegetarian"))

    clean = cand[(cand["total_min"] > 0) & (cand["ingredients"].map(len) >= 3) & (cand["steps"].map(len) >= 2)
                 & (cand["ReviewCount"].fillna(0) >= 3)].reset_index(drop=True)
    print("after quality filter:\n", clean["dish"].value_counts())

    rng = np.random.RandomState(config.SEED)
    chosen = []
    for _, g in clean.groupby("dish"):
        pick = set()
        for sub, k in [(g[g.is_vegan], 4), (g[g.is_vegetarian], 6), (g[g.total_min <= 30], 6), (g, 20)]:
            pool = sub.index.difference(list(pick))
            if len(pool):
                pick.update(rng.choice(pool, size=min(k, len(pool)), replace=False).tolist())
        chosen.append(g.loc[sorted(pick)])
    corpus = pd.concat(chosen)

    out = pd.DataFrame({
        "recipe_id": corpus["RecipeId"].astype(int),
        "title": corpus["Name"],
        "dish": corpus["dish"],
        "category": corpus["RecipeCategory"],
        "keywords": corpus["keywords"].map(lambda x: "; ".join(x)),
        "ingredients": corpus["ingredients"].map(lambda x: "; ".join(x)),
        "instructions": corpus["steps"].map(lambda x: " ".join(x)),
        "total_min": corpus["total_min"].astype(int),
        "is_vegetarian": corpus["is_vegetarian"],
        "is_vegan": corpus["is_vegan"],
        "rating": corpus["AggregatedRating"],
        "reviews": corpus["ReviewCount"].fillna(0).astype(int),
    }).reset_index(drop=True)

    config.DATA_DIR.mkdir(exist_ok=True)
    out.to_csv(config.DATA_DIR / "recipes.csv", index=False)
    print("recipes in corpus:", len(out))
    print(out["dish"].value_counts().to_string())


if __name__ == "__main__":
    main()
