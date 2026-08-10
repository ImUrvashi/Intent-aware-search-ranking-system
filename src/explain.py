from __future__ import annotations

from typing import List

import pandas as pd

_STRONG = 0.66
_MEDIUM = 0.33


def _text_match_reason(cosine: float) -> str | None:
    if cosine >= 0.60:
        return "Strong match for your search"
    if cosine > 0.20:
        return "Matches some of your search terms"
    return None


def explain_row(row: pd.Series) -> List[str]:
    reasons: List[str] = []

    match = _text_match_reason(float(row.get("cosine_similarity", 0.0)))
    if match:
        reasons.append(match)

    rating_norm = float(row.get("rating_norm", 0.0))
    if rating_norm >= _STRONG:
        star = row.get("rating")
        reasons.append(
            f"Highly rated ({star:.1f}/5)" if pd.notna(star) else "Highly rated"
        )

    price_score = float(row.get("price_score", 0.0))
    if price_score >= _STRONG:
        reasons.append("Budget-friendly price")
    elif price_score <= _MEDIUM:
        reasons.append("Premium price")

    if float(row.get("review_score", 0.0)) >= _STRONG:
        reasons.append("Backed by many reviews")

    if float(row.get("sentiment_score", 0.0)) >= _STRONG:
        reasons.append("Buyers are very positive")

    if float(row.get("popularity_score", 0.0)) >= _STRONG:
        reasons.append("Popular choice")

    if not reasons:
        reasons.append("A reasonable overall match")

    return reasons


def explain_text(row: pd.Series, separator: str = " · ") -> str:
    return separator.join(explain_row(row))


def top_drivers(ranker, n: int = 3) -> List[str]:
    from src import model as _model

    importances = _model.feature_importances(ranker)
    return list(importances.head(n).index)


def main() -> None:
    from src import ranking

    engine = ranking.load_engine()
    query = "moisture wicking golf polo"
    results = ranking.rank_products(query, engine, top_k=5)

    print(f"Query: '{query}'\n")
    for _, row in results.iterrows():
        print(f"- {row['title'][:65]}")
        print(f"    why: {explain_text(row)}\n")


if __name__ == "__main__":
    main()
