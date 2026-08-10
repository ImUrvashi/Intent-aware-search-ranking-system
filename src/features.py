from __future__ import annotations

import os
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

CLEANED_CSV = os.path.join(DATA_DIR, "cleaned_products.csv")
VECTORIZER_PKL = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
MATRIX_PKL = os.path.join(MODELS_DIR, "product_matrix.pkl")
INDEX_PKL = os.path.join(MODELS_DIR, "product_index.pkl")

# Query-independent scores, computed once and reused across every search.
STATIC_FEATURE_COLUMNS = [
    "rating_norm",
    "price_score",
    "review_score",
    "sentiment_score",
    "popularity_score",
]

FEATURE_COLUMNS = ["cosine_similarity"] + STATIC_FEATURE_COLUMNS

# Raw cosine below this is treated as noise, not a real match — see PROJECT_GUIDE.md §3 Step 7.
_RAW_COSINE_FLOOR = 0.03


def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
    )


def fit_product_matrix(products: pd.DataFrame):
    if "search_text" not in products.columns:
        raise KeyError("Expected a 'search_text' column in the cleaned products table.")

    text = products["search_text"].fillna("").astype(str)
    vectorizer = build_vectorizer()
    product_matrix = vectorizer.fit_transform(text)
    return vectorizer, product_matrix


def embed_query(query: str, vectorizer: TfidfVectorizer):
    if query is None:
        query = ""
    return vectorizer.transform([str(query)])


def query_similarity(query: str, vectorizer: TfidfVectorizer, product_matrix) -> np.ndarray:
    query_vec = embed_query(query, vectorizer)
    scores = cosine_similarity(query_vec, product_matrix).ravel()
    return scores


def search(
    query: str,
    products: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    product_matrix,
    top_k: int = 5,
) -> pd.DataFrame:
    scores = query_similarity(query, vectorizer, product_matrix)

    results = products.copy()
    results["similarity"] = scores

    columns = ["title", "brand", "category", "price", "rating", "similarity"]
    columns = [c for c in columns if c in results.columns]
    return results.sort_values("similarity", ascending=False).head(top_k)[columns]


def _minmax(series: pd.Series) -> pd.Series:
    low, high = series.min(), series.max()
    if high == low:
        return pd.Series(0.5, index=series.index)
    return (series - low) / (high - low)


def _inverted_minmax(series: pd.Series) -> pd.Series:
    return 1.0 - _minmax(series)


def _log_minmax(series: pd.Series) -> pd.Series:
    return _minmax(np.log1p(series.clip(lower=0)))


def add_static_features(products: pd.DataFrame) -> pd.DataFrame:
    df = products.copy()

    df["rating_norm"] = (df["rating"] / 5.0).clip(0.0, 1.0)
    df["price_score"] = _inverted_minmax(df["price"])
    df["review_score"] = _log_minmax(df["review_count"])
    df["sentiment_score"] = ((df["avg_sentiment"] + 1.0) / 2.0).clip(0.0, 1.0)
    df["popularity_score"] = 1.0 - _log_minmax(df["bestseller_rank"])

    return df


# Intent guard — see PROJECT_GUIDE.md §3 Step 7 for the full rationale.
# Two hand-maintained maps; everything else is read live from the catalog.
_INTENT_SYNONYMS = {
    "jeans":      ["jeans", "jean", "denim", "denims"],
    "sweatpants": ["sweatpants", "sweatpant", "jogger", "joggers", "track pant", "track pants", "trackpants"],
    "shorts":     ["shorts"],
    "polo":       ["polo", "polos"],
    "t-shirt":    ["t-shirt", "t shirt", "tshirt", "tshirts", "tee", "tees", "crewneck", "crew neck"],
    "shirt":      ["button-down", "button down", "dress shirt", "casual shirt", "flannel", "blouse", "oxford shirt"],
    "underwear":  ["boxer", "boxers", "briefs", "brief", "underwear", "undershirt", "undershirts"],
    "sneakers":   ["sneaker", "sneakers", "trainer", "trainers", "kicks", "plimsoll"],
    "running":    ["running", "running shoe", "running shoes", "runner", "runners", "road running", "jogging shoe"],
    "shoes":      ["shoe", "shoes", "footwear", "loafer", "loafers", "boot", "boots"],
    "chinos":     ["chino", "chinos", "trouser", "trousers", "dress pant", "dress pants", "slacks"],
    "hoodie":     ["hoodie", "hoodies", "sweatshirt", "sweatshirts"],
    "jacket":     ["jacket", "jackets", "coat", "coats"],
}

# For types whose name isn't literally a substring of any real category label.
_TYPE_TO_CATEGORY_HINT = {
    "underwear": ["boxer", "brief", "underwear", "undershirt"],
    "running":   ["running", "road"],
    "shoes":     ["sneaker", "running", "loafer", "boot", "shoe", "footwear"],
    "chinos":    ["chino", "casual", "pant", "trouser"],
    "t-shirt":   ["t-shirt", "tee", "shirt"],
    "shirt":     ["blouse", "button", "casual", "shirt"],
    "jacket":    ["jacket", "coat"],
    "hoodie":    ["hoodie", "sweatshirt"],
}

_INTENT_PATTERNS = {
    type_name: re.compile(
        r"\b(?:" + "|".join(re.escape(s) for s in synonyms) + r")\b"
    )
    for type_name, synonyms in _INTENT_SYNONYMS.items()
}


def detect_intent_types(query: str) -> set:
    q = str(query).lower()
    return {name for name, pat in _INTENT_PATTERNS.items() if pat.search(q)}


def detect_intent_categories(query: str, products: pd.DataFrame) -> set:
    types = detect_intent_types(query)
    if not types or "category" not in products.columns:
        return set()

    labels = {str(c).lower() for c in products["category"].dropna().unique()}
    required: set = set()
    for type_name in types:
        hints = _TYPE_TO_CATEGORY_HINT.get(type_name, [type_name])
        for label in labels:
            if any(hint in label for hint in hints):
                required.add(label)
    return required


def intent_match_mask(query: str, products: pd.DataFrame):
    """None means "don't filter" — no type detected, or nothing in the catalog matches it."""
    if "category" not in products.columns:
        return None

    required = detect_intent_categories(query, products)
    if not required:
        return None

    matches = products["category"].fillna("").str.lower().isin(required)
    if not matches.any():
        return None
    return matches


def query_feature_matrix(
    query: str,
    products_static: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    product_matrix,
) -> pd.DataFrame:
    cosine = query_similarity(query, vectorizer, product_matrix)
    cosine = np.where(cosine < _RAW_COSINE_FLOOR, 0.0, cosine)

    mask = intent_match_mask(query, products_static)
    if mask is not None:
        cosine = cosine * mask.to_numpy().astype(float)

    # Raw cosine is tiny (~0.0-0.3) next to the 0..1 static scores, so it's
    # rescaled per-query (best match -> 1.0) or quality would drown out intent.
    cosine = _minmax(pd.Series(cosine, index=products_static.index)).to_numpy()

    feats = pd.DataFrame(index=products_static.index)
    feats["cosine_similarity"] = cosine
    for col in STATIC_FEATURE_COLUMNS:
        feats[col] = products_static[col].to_numpy()

    return feats[FEATURE_COLUMNS]


def save_artifacts(vectorizer: TfidfVectorizer, product_matrix, products: pd.DataFrame) -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)

    joblib.dump(vectorizer, VECTORIZER_PKL)
    joblib.dump(product_matrix, MATRIX_PKL)

    id_cols = [c for c in ["asin", "title"] if c in products.columns]
    joblib.dump(products[id_cols].reset_index(drop=True), INDEX_PKL)


def load_artifacts():
    missing = [p for p in (VECTORIZER_PKL, MATRIX_PKL, INDEX_PKL) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Missing Phase 3 artifacts: "
            + ", ".join(os.path.basename(m) for m in missing)
            + ". Run `python -m src.features` first."
        )
    vectorizer = joblib.load(VECTORIZER_PKL)
    product_matrix = joblib.load(MATRIX_PKL)
    product_index = joblib.load(INDEX_PKL)
    return vectorizer, product_matrix, product_index


def main() -> None:
    print("Loading cleaned products from:", CLEANED_CSV)
    products = pd.read_csv(CLEANED_CSV)
    print(f"  -> {len(products)} products loaded.\n")

    print("Fitting TF-IDF vectorizer on 'search_text' ...")
    vectorizer, product_matrix = fit_product_matrix(products)
    print(f"  -> vocabulary size : {len(vectorizer.vocabulary_):,} terms")
    print(f"  -> product matrix  : {product_matrix.shape[0]} rows x "
          f"{product_matrix.shape[1]} columns\n")

    print("Saving artifacts to models/ ...")
    save_artifacts(vectorizer, product_matrix, products)
    print(f"  -> {os.path.relpath(VECTORIZER_PKL, PROJECT_ROOT)}")
    print(f"  -> {os.path.relpath(MATRIX_PKL, PROJECT_ROOT)}")
    print(f"  -> {os.path.relpath(INDEX_PKL, PROJECT_ROOT)}\n")

    demo_query = "moisture wicking golf polo"
    print(f"Demo search for: '{demo_query}'")
    top = search(demo_query, products, vectorizer, product_matrix, top_k=5)
    with pd.option_context("display.max_colwidth", 60, "display.width", 120):
        print(top.to_string(index=False))


if __name__ == "__main__":
    main()
