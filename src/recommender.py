from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src import features


def _row_position(products: pd.DataFrame, asin: str) -> int:
    """Positional row number, so it lines up with the matching row of product_matrix."""
    matches = np.where(products["asin"].to_numpy() == asin)[0]
    if len(matches) == 0:
        raise ValueError(f"Product asin '{asin}' was not found in the products table.")
    return int(matches[0])


def similar_products(
    asin: str,
    products: pd.DataFrame,
    product_matrix,
    top_k: int = 5,
) -> pd.DataFrame:
    position = _row_position(products, asin)

    sims = cosine_similarity(product_matrix[position], product_matrix).ravel()
    sims[position] = -1.0  # never recommend the product to itself

    best = np.argsort(sims)[::-1][:top_k]

    result = products.iloc[best].copy()
    result["similarity"] = sims[best]

    columns = ["asin", "title", "brand", "category", "price", "rating",
               "image_url", "product_url", "similarity"]
    columns = [c for c in columns if c in result.columns]
    return result[columns].reset_index(drop=True)


def main() -> None:
    products = pd.read_csv(features.CLEANED_CSV)
    _, product_matrix, _ = features.load_artifacts()

    seed_asin = products.iloc[0]["asin"]
    print("Seed product:", products.iloc[0]["title"][:70], "...\n")
    print("You may also like:")
    recs = similar_products(seed_asin, products, product_matrix, top_k=5)
    with pd.option_context("display.max_colwidth", 60, "display.width", 120):
        print(recs[["title", "category", "price", "rating", "similarity"]].to_string(index=False))


if __name__ == "__main__":
    main()
