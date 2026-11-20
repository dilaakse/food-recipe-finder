"""Create the 30 test queries and their relevant recipes (data/queries.csv, data/qrels.csv).

Run:  python -m scripts.build_queries

Relevance is rule-based: a recipe is relevant to a query when it is the right dish and
satisfies the diet, time limit and required ingredient / tag / instruction words of that
query. The rules only look at recipe metadata and text, never at model outputs.
Caveat: the diet flags and times used here are the same metadata that the optional
regex filter of the search pipeline uses, so results "with metadata filter" are optimistic.
"""
import pandas as pd

from src import config

# (id, query, dish, diet, max_minutes, ingredient_regex, tag_regex, instruction_regex)
Q = [
    (1,  "pizza recipe",                       "pizza",          None,         None, None, None, None),
    (2,  "vegetarian pizza",                   "pizza",          "vegetarian", None, None, None, None),
    (3,  "lasagna recipe",                     "lasagna",        None,         None, None, None, None),
    (4,  "vegan chocolate cake",               "chocolate_cake", "vegan",      None, None, None, None),
    (5,  "omelette recipe",                    "omelette",       None,         None, None, None, None),
    (6,  "quick omelette",                     "omelette",       None,         30,   None, None, None),
    (7,  "easy tacos recipe",                  "tacos",          None,         None, None, r"\beasy\b", None),
    (8,  "quick pizza under 30 minutes",       "pizza",          None,         30,   None, None, None),
    (9,  "piza recipe",                        "pizza",          None,         None, None, None, None),
    (10, "classic hamburger recipe",           "hamburger",      None,         None, None, None, None),
    (11, "juicy beef burger on the grill",     "hamburger",      None,         None, r"beef", None, r"grill"),
    (12, "vegetarian burger",                  "hamburger",      "vegetarian", None, None, None, None),
    (13, "fluffy pancakes",                    "pancakes",       None,         None, None, None, None),
    (14, "pancakes under 20 minutes",          "pancakes",       None,         20,   None, None, None),
    (15, "vegan pancakes",                     "pancakes",       "vegan",      None, None, None, None),
    (16, "homemade vanilla ice cream",         "ice_cream",      None,         None, r"vanilla", None, None),
    (17, "chocolate ice cream",                "ice_cream",      None,         None, r"chocolate|cocoa", None, None),
    (18, "dairy free ice cream",               "ice_cream",      None,         None, None, r"dairy[- ]free|lactose|vegan", None),
    (19, "something sweet and cold",           "ice_cream",      None,         None, None, None, None),
    (20, "crispy french fries",                "french_fries",   None,         None, None, None, None),
    (21, "oven baked french fries",            "french_fries",   None,         None, None, None, r"oven|bake"),
    (22, "omelette with mushrooms",            "omelette",       None,         None, r"mushroom", None, None),
    (23, "easy chocolate cake",                "chocolate_cake", None,         None, None, r"\beasy\b", None),
    (24, "vegetarian lasagna",                 "lasagna",        "vegetarian", None, None, None, None),
    (25, "lasagna with ground beef",           "lasagna",        None,         None, r"ground beef|ground chuck|lean ground", None, None),
    (26, "chicken tacos",                      "tacos",          None,         None, r"chicken", None, None),
    (27, "vegetarian tacos",                   "tacos",          "vegetarian", None, None, None, None),
    (28, "quick dinner with tortillas",        "tacos",          None,         30,   r"tortilla", None, None),
    (29, "caesar salad",                       "caesar_salad",   None,         None, None, None, None),
    (30, "caesar salad with chicken",          "caesar_salad",   None,         None, r"chicken", None, None),
]


def main():
    recipes = pd.read_csv(config.DATA_DIR / "recipes.csv")
    for c in ["title", "category", "keywords", "ingredients", "instructions"]:
        recipes[c] = recipes[c].fillna("")
    recipes["ing_title"] = (recipes["title"] + " " + recipes["ingredients"]).str.lower()
    recipes["tags"] = (recipes["title"] + " " + recipes["keywords"] + " " + recipes["category"]).str.lower()
    recipes["instr"] = recipes["instructions"].str.lower()

    def relevant_ids(dish, diet, max_min, ing, tag, instr):
        m = recipes["dish"] == dish
        if diet == "vegetarian":
            m &= recipes["is_vegetarian"]
        if diet == "vegan":
            m &= recipes["is_vegan"]
        if max_min:
            m &= recipes["total_min"] <= max_min
        if ing:
            m &= recipes["ing_title"].str.contains(ing, regex=True)
        if tag:
            m &= recipes["tags"].str.contains(tag, regex=True)
        if instr:
            m &= recipes["instr"].str.contains(instr, regex=True)
        return recipes.loc[m, "recipe_id"].tolist()

    queries, qrels = [], []
    for qid, text, *spec in Q:
        ids = relevant_ids(*spec)
        queries.append(dict(query_id=qid, query=text, dish=spec[0]))
        qrels += [dict(query_id=qid, recipe_id=i) for i in ids]

    queries, qrels = pd.DataFrame(queries), pd.DataFrame(qrels)
    queries["n_relevant"] = queries["query_id"].map(qrels.groupby("query_id").size()).fillna(0).astype(int)
    queries.drop(columns="n_relevant").to_csv(config.DATA_DIR / "queries.csv", index=False)
    qrels.to_csv(config.DATA_DIR / "qrels.csv", index=False)
    print(queries[["query_id", "query", "n_relevant"]].to_string(index=False))


if __name__ == "__main__":
    main()
