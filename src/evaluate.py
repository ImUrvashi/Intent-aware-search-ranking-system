from __future__ import annotations

import numpy as np
import pandas as pd

from src import ranking

# Different phrasing from TRAINING_QUERIES in src/model.py, to test generalisation.
TEST_QUERIES = [
    ("breathable golf polo shirt", "Polos"),
    ("straight leg denim jeans", "Jeans"),
    ("everyday cotton t shirt", "T-Shirts"),
    ("athletic jogger sweatpants", "Sweatpants"),
    ("running trainers for the gym", "Road Running"),
    ("comfy boxer briefs underwear", "Boxer Briefs"),
    ("casual fashion sneakers", "Fashion Sneakers"),
    ("active gym workout shorts", "Active Shorts"),
]


def precision_at_k(relevant_flags: list[int], k: int) -> float:
    top = relevant_flags[:k]
    return sum(top) / k if k else 0.0


def ndcg_at_k(relevant_flags: list[int], k: int) -> float:
    rel = np.asarray(relevant_flags[:k], dtype=float)
    if rel.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rel.size + 2))
    dcg = float(np.sum(rel * discounts))

    ideal = np.sort(rel)[::-1]
    idcg = float(np.sum(ideal * discounts))
    return dcg / idcg if idcg > 0 else 0.0


def _relevance_flags(results: pd.DataFrame, expected_category: str) -> list[int]:
    return [int(cat == expected_category) for cat in results["category"].tolist()]


def _quality_gains(results: pd.DataFrame, expected_category: str) -> list[float]:
    """Graded relevance: an on-topic product's gain is its star rating, else 0."""
    gains = []
    for _, row in results.iterrows():
        if row["category"] == expected_category and pd.notna(row.get("rating")):
            gains.append(float(row["rating"]))
        else:
            gains.append(0.0)
    return gains


def evaluate_ranker(rank_fn, engine, k: int = 5) -> dict:
    precisions, ndcgs, q_ndcgs = [], [], []
    for query, expected_category in TEST_QUERIES:
        results = rank_fn(query, engine, top_k=k)
        flags = _relevance_flags(results, expected_category)
        precisions.append(precision_at_k(flags, k))
        ndcgs.append(ndcg_at_k(flags, k))
        q_ndcgs.append(ndcg_at_k(_quality_gains(results, expected_category), k))

    return {
        f"precision@{k}": float(np.mean(precisions)),
        f"ndcg@{k}": float(np.mean(ndcgs)),
        f"quality_ndcg@{k}": float(np.mean(q_ndcgs)),
    }


def main(k: int = 5) -> None:
    engine = ranking.load_engine()

    baseline = evaluate_ranker(ranking.rank_by_cosine, engine, k=k)
    ml_model = evaluate_ranker(ranking.rank_products, engine, k=k)

    print(f"Evaluation over {len(TEST_QUERIES)} held-out queries (k={k})\n")
    header = f"{'Metric':<20}{'Cosine-only (A)':>18}{'ML ranker (B)':>16}{'Winner':>10}"
    print(header)
    print("-" * len(header))
    for metric in (f"precision@{k}", f"ndcg@{k}", f"quality_ndcg@{k}"):
        a, b = baseline[metric], ml_model[metric]
        winner = "B (ML)" if b > a else ("A (cos)" if a > b else "tie")
        print(f"{metric:<20}{a:>18.3f}{b:>16.3f}{winner:>10}")

    print(
        "\nReading the table:\n"
        "  * precision@k / ndcg@k  -> 'did we return the right TYPE of product?'\n"
        "    Both rankers do well here, because both rely on the text match.\n"
        "  * quality_ndcg@k        -> 'did we put the BEST on-topic products first?'\n"
        "    This is where the ML ranker adds value over plain keyword search by\n"
        "    blending in rating, price, reviews and popularity."
    )


if __name__ == "__main__":
    main()
