from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src import features

TRAINING_CSV = os.path.join(features.DATA_DIR, "training_data.csv")
RANKER_PKL = os.path.join(features.MODELS_DIR, "ranker.pkl")

# Used only to generate training examples; the trained model works on any query.
TRAINING_QUERIES = [
    "moisture wicking golf polo",
    "slim fit jeans for work",
    "comfortable running shoes",
    "cotton button down shirt",
    "casual short sleeve t shirt",
    "warm winter jacket",
    "breathable athletic shorts",
    "formal dress shirt for office",
    "stretchy skinny jeans",
    "lightweight summer shirt",
    "leather dress shoes",
    "soft cotton boxer briefs",
    "waterproof hiking boots",
    "classic denim jacket",
    "quick dry swim shorts",
]

# relevance = text_match * (0.5 + 0.5 * quality) — multiplicative gate, see
# PROJECT_GUIDE.md §3 Step 5 for why this isn't a plain weighted sum.
QUALITY_WEIGHTS = {
    "rating_norm": 0.50,
    "review_score": 0.20,
    "popularity_score": 0.15,
    "price_score": 0.15,
    "sentiment_score": 0.00,  # already reflected via review_score; kept as a model input
}


def _true_relevance(feats: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    text_match = feats["cosine_similarity"].to_numpy()

    quality = np.zeros(len(feats), dtype=float)
    for col, weight in QUALITY_WEIGHTS.items():
        quality += weight * feats[col].to_numpy()

    score = text_match * (0.5 + 0.5 * quality)
    score += rng.normal(loc=0.0, scale=0.01, size=len(feats))
    return score


def build_training_data(
    products: pd.DataFrame,
    vectorizer,
    product_matrix,
    n_positive: int = 10,
    n_negative: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    products_static = features.add_static_features(products)

    rows = []
    for query in TRAINING_QUERIES:
        feats = features.query_feature_matrix(
            query, products_static, vectorizer, product_matrix
        )
        relevance = _true_relevance(feats, rng)

        order = np.argsort(relevance)[::-1]
        positive_idx = order[:n_positive]
        negative_pool = order[n_positive:]
        take = min(n_negative, len(negative_pool))
        negative_idx = rng.choice(negative_pool, size=take, replace=False)

        for idx in list(positive_idx) + list(negative_idx):
            label = 1 if idx in positive_idx else 0
            row = {
                "query": query,
                "asin": products.iloc[idx]["asin"],
                "title": products.iloc[idx]["title"],
            }
            for col in features.FEATURE_COLUMNS:
                row[col] = float(feats.iloc[idx][col])
            row["relevance"] = label
            rows.append(row)

    return pd.DataFrame(rows)


def save_training_data(df: pd.DataFrame) -> None:
    os.makedirs(features.DATA_DIR, exist_ok=True)
    df.to_csv(TRAINING_CSV, index=False)


def train_ranker(training_df: pd.DataFrame, seed: int = 42):
    X = training_df[features.FEATURE_COLUMNS].to_numpy()
    y = training_df["relevance"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=seed,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    return model, metrics


def feature_importances(model) -> pd.Series:
    importances = pd.Series(
        model.feature_importances_, index=features.FEATURE_COLUMNS
    )
    return importances.sort_values(ascending=False)


def save_ranker(model) -> None:
    os.makedirs(features.MODELS_DIR, exist_ok=True)
    joblib.dump(model, RANKER_PKL)


def load_ranker():
    if not os.path.exists(RANKER_PKL):
        raise FileNotFoundError(
            "Missing models/ranker.pkl. Run `python -m src.model` first."
        )
    return joblib.load(RANKER_PKL)


def main() -> None:
    print("Loading cleaned products + Phase 3 artifacts ...")
    products = pd.read_csv(features.CLEANED_CSV)
    vectorizer, product_matrix, _ = features.load_artifacts()

    print(f"Building training data for {len(TRAINING_QUERIES)} queries ...")
    training_df = build_training_data(products, vectorizer, product_matrix)
    save_training_data(training_df)
    pos = int(training_df["relevance"].sum())
    print(f"  -> {len(training_df)} examples ({pos} relevant, "
          f"{len(training_df) - pos} not). Saved to "
          f"{os.path.relpath(TRAINING_CSV, features.PROJECT_ROOT)}\n")

    print("Training the RandomForest ranker ...")
    model, metrics = train_ranker(training_df)
    save_ranker(model)
    print(f"  -> Saved {os.path.relpath(RANKER_PKL, features.PROJECT_ROOT)}\n")

    print("Test-set performance:")
    for name in ("accuracy", "precision", "recall", "roc_auc"):
        print(f"  {name:<10}: {metrics[name]:.3f}")

    print("\nFeature importances (what drives relevance):")
    for name, value in feature_importances(model).items():
        print(f"  {name:<18}: {value:.3f}")


if __name__ == "__main__":
    main()
