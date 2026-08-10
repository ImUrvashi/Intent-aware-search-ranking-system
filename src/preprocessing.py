from __future__ import annotations

import ast
import os
import re

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
RAW_DIR = os.path.join(PROJECT_ROOT, "Amazon-Ecom_dataset")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PRODUCTS_CSV = os.path.join(RAW_DIR, "products.csv")
REVIEWS_CSV = os.path.join(RAW_DIR, "reviews.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "cleaned_products.csv")


def parse_rating(value) -> float:
    if pd.isna(value):
        return np.nan
    match = re.search(r"([0-9]+\.?[0-9]*)", str(value))
    return float(match.group(1)) if match else np.nan


def parse_review_count(value) -> float:
    if pd.isna(value):
        return np.nan
    digits = re.sub(r"[^0-9]", "", str(value))
    return float(digits) if digits else np.nan


def parse_bestseller_rank(value) -> float:
    """Take the LAST '#<number>' — it's the specific sub-category rank, not the overall one."""
    if pd.isna(value):
        return np.nan
    matches = re.findall(r"#\s*([0-9,]+)", str(value))
    if not matches:
        return np.nan
    last = matches[-1].replace(",", "")
    return float(last) if last else np.nan


def extract_category(breadcrumbs) -> str:
    if pd.isna(breadcrumbs):
        return "Unknown"
    text = str(breadcrumbs).replace(">", "›")
    parts = [p.strip() for p in text.split("›") if p.strip()]
    return parts[-1] if parts else "Unknown"


def first_image_url(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)) and parsed:
            return str(parsed[0])
    except (ValueError, SyntaxError):
        pass
    match = re.search(r"https?://[^\s'\"\]]+", text)
    return match.group(0) if match else ""


def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def aggregate_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    grouped = reviews.groupby("productASIN").agg(
        avg_sentiment=("sentiment_score", "mean"),
        n_reviews=("sentiment_score", "size"),
    )
    grouped = grouped.reset_index().rename(columns={"productASIN": "asin"})
    return grouped


def build_clean_products(
    products_path: str = PRODUCTS_CSV,
    reviews_path: str = REVIEWS_CSV,
) -> pd.DataFrame:
    products = pd.read_csv(products_path)
    reviews = pd.read_csv(reviews_path)

    out = pd.DataFrame()
    out["asin"] = products["asin"]
    out["title"] = products["title"].fillna("").astype(str)
    out["brand"] = products["brand_name"].fillna("Unknown").astype(str)
    out["category"] = products["breadcrumbs"].apply(extract_category)
    out["image_url"] = products["all_images"].apply(first_image_url)
    out["product_url"] = products["product_url"].fillna("").astype(str)

    # product_description is empty ~63% of the time, so search_text relies mainly
    # on title + about_item and only adds description when present.
    combined = (
        products["title"].fillna("")
        + " "
        + products["about_item"].fillna("")
        + " "
        + products["product_description"].fillna("")
    )
    out["search_text"] = combined.apply(clean_text)

    out["price"] = pd.to_numeric(products["price_value"], errors="coerce")
    out["rating"] = products["rating_stars"].apply(parse_rating)
    out["review_count"] = products["rating_count"].apply(parse_review_count)
    out["bestseller_rank"] = products["best_sellers_rank"].apply(parse_bestseller_rank)

    review_stats = aggregate_reviews(reviews)
    out = out.merge(review_stats, on="asin", how="left")

    out = _fill_missing(out)
    out = _drop_unusable_rows(out)
    return out


def _fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["price"] = df["price"].fillna(df["price"].median())
    df["rating"] = df["rating"].fillna(df["rating"].median())
    df["review_count"] = df["review_count"].fillna(0)

    worst_rank = df["bestseller_rank"].max()
    worst_rank = (worst_rank + 1) if pd.notna(worst_rank) else 999_999
    df["bestseller_rank"] = df["bestseller_rank"].fillna(worst_rank)

    df["avg_sentiment"] = df["avg_sentiment"].fillna(0.0)
    df["n_reviews"] = df["n_reviews"].fillna(0).astype(int)
    return df


def _drop_unusable_rows(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df[df["search_text"].str.strip() != ""].copy()
    df = df.drop_duplicates(subset="asin").reset_index(drop=True)
    removed = before - len(df)
    if removed:
        print(f"Removed {removed} unusable/duplicate rows.")
    return df


def save_clean_products(df: pd.DataFrame, output_path: str = OUTPUT_CSV) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} cleaned products -> {output_path}")


def main() -> None:
    df = build_clean_products()
    save_clean_products(df)

    print("\n--- Cleaned data summary ---")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print("\nNulls per column:")
    print(df.isna().sum())
    print("\nFirst 3 rows (key columns):")
    print(
        df[["asin", "category", "price", "rating", "review_count", "avg_sentiment"]]
        .head(3)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
