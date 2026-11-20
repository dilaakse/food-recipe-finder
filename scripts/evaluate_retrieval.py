"""Evaluate the three retrieval models on the 30 test queries.

Run:  python -m scripts.evaluate_retrieval

Main comparison: text only (diet / time words stay in the query as plain text). Ablation: the
same models with the regex metadata filter (optimistic, because the relevance labels use the
same metadata). Latency is CPU time for one query, averaged over 100 runs.
Writes results/retrieval_results.json and results/retrieval_per_query.csv.
"""
import pandas as pd

from src import config
from src.benchmark import cpu_name, query_latency_ms
from src.retrieval import (DenseRetriever, TfidfRetriever, evaluate, load_eval_data,
                           load_recipes, per_query)
from src.text_pipeline import QueryProcessor, apply_filters, extract_constraints


def main():
    recipes = load_recipes()
    queries, rel = load_eval_data()
    qp = QueryProcessor(recipes["doc"])

    models = [TfidfRetriever(recipes, qp), DenseRetriever(recipes, qp, quantize=False),
              DenseRetriever(recipes, qp, quantize=True)]
    labels = [("TF-IDF + cosine (stemmed)", "n/a"), ("MiniLM embeddings (BERT-style)", "fp32"),
              ("MiniLM embeddings (BERT-style)", "int8")]

    rows = []
    for m, (name, prec) in zip(models, labels):
        rows.append(dict(model=name, precision=prec, **evaluate(m.scores, recipes, queries, rel),
                         size_mb=round(m.size_mb(), 2), latency_ms=round(query_latency_ms(m.scores), 2)))
    df = pd.DataFrame(rows)
    print("CPU:", cpu_name())
    print(df.to_string(index=False))

    def filtered(m):
        return lambda q: apply_filters(m.scores(q), extract_constraints(q), recipes)

    print("\nwith metadata filter:")
    for m in models:
        print(m.name, evaluate(filtered(m), recipes, queries, rel))

    config.RESULTS_DIR.mkdir(exist_ok=True)
    df.to_json(config.RESULTS_DIR / "retrieval_results.json", orient="records", indent=2)

    a, b = per_query(models[0].scores, recipes, queries, rel), per_query(models[2].scores, recipes, queries, rel)
    t = queries[["query_id", "query"]].copy()
    t["tfidf_rank"], t["tfidf_top1"] = [x[0] for x in a], [x[1] for x in a]
    t["int8_rank"], t["int8_top1"] = [x[0] for x in b], [x[1] for x in b]
    t.to_csv(config.RESULTS_DIR / "retrieval_per_query.csv", index=False)


if __name__ == "__main__":
    main()
