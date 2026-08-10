from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src import features, model


@dataclass
class SearchEngine:
    products: pd.DataFrame
    vectorizer: object
    product_matrix: object
    ranker: object


def load_engine() -> SearchEngine:
    products = pd.read_csv(features.CLEANED_CSV)
    products_static = features.add_static_features(products)
    vectorizer, product_matrix, _ = features.load_artifacts()
    ranker = model.load_ranker()
    return SearchEngine(products_static, vectorizer, product_matrix, ranker)


_DISPLAY_COLUMNS = [
    "asin", "title", "brand", "category", "price", "rating",
    "review_count", "image_url", "product_url", "score",
]


def _assemble_results(engine: SearchEngine, feats: pd.DataFrame, scores, top_k: int) -> pd.DataFrame:
    out = engine.products.copy()
    out["score"] = scores
    for col in features.FEATURE_COLUMNS:
        out[col] = feats[col].to_numpy()

    out = out.sort_values("score", ascending=False).head(top_k).reset_index(drop=True)
    keep = [c for c in _DISPLAY_COLUMNS if c in out.columns] + features.FEATURE_COLUMNS
    return out[keep]


def _apply_intent_guard(query: str, engine: SearchEngine, scores):
    """Zero the final score for off-type products too, not just the text-match input."""
    mask = features.intent_match_mask(query, engine.products)
    if mask is None:
        return scores
    return scores * mask.to_numpy().astype(float)


def rank_products(query: str, engine: SearchEngine, top_k: int = 10) -> pd.DataFrame:
    feats = features.query_feature_matrix(
        query, engine.products, engine.vectorizer, engine.product_matrix
    )
    scores = engine.ranker.predict_proba(feats[features.FEATURE_COLUMNS].to_numpy())[:, 1]
    scores = _apply_intent_guard(query, engine, scores)
    return _assemble_results(engine, feats, scores, top_k)


def rank_by_cosine(query: str, engine: SearchEngine, top_k: int = 10) -> pd.DataFrame:
    """Text-match-only ranking — the keyword baseline for the evaluation in src/evaluate.py."""
    feats = features.query_feature_matrix(
        query, engine.products, engine.vectorizer, engine.product_matrix
    )
    scores = feats["cosine_similarity"].to_numpy()
    return _assemble_results(engine, feats, scores, top_k)


def main() -> None:
    engine = load_engine()
    query = "moisture wicking golf polo"

    print(f"Query: '{query}'\n")
    print("=== ML ranking (smart) ===")
    cols = ["title", "category", "price", "rating", "score"]
    with pd.option_context("display.max_colwidth", 55, "display.width", 120):
        print(rank_products(query, engine, top_k=5)[cols].to_string(index=False))
        print("\n=== Cosine-only ranking (baseline) ===")
        print(rank_by_cosine(query, engine, top_k=5)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
