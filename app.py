from __future__ import annotations

import pandas as pd
import streamlit as st

from src import explain, ranking, recommender

_EXAMPLES = [
    "moisture wicking golf polo",
    "slim fit jeans for work",
    "comfortable running shoes",
    "warm sweatpants for winter",
]

_REASON_COLOURS = {
    "match": "green",
    "rated": "orange",
    "price": "blue",
    "reviews": "violet",
    "positive": "green",
    "popular": "red",
}


@st.cache_resource(show_spinner="Loading search engine …")
def get_engine() -> ranking.SearchEngine:
    return ranking.load_engine()


def _format_price(value) -> str:
    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return "—"


def _reason_colour(reason: str) -> str:
    low = reason.lower()
    for key, colour in _REASON_COLOURS.items():
        if key in low:
            return colour
    return "gray"


def _has_image(url) -> bool:
    return isinstance(url, str) and url.startswith("http")


def _set_query(text: str) -> None:
    """Runs before the script reruns — the only safe time to write to a widget's session-state key."""
    st.session_state["query"] = text


def _render_product(row: pd.Series, rank: int, *, show_reason: bool) -> None:
    with st.container(border=True):
        image_col, info_col = st.columns([1, 4], vertical_alignment="center")

        with image_col:
            if _has_image(row.get("image_url")):
                st.image(row["image_url"], width=90)

        with info_col:
            title = row.get("title", "Untitled product")
            url = row.get("product_url")
            title_md = f"[{title}]({url})" if _has_image(url) else title
            st.markdown(f"**#{rank} · {title_md}**")

            rating = row.get("rating")
            reviews = row.get("review_count")
            bits = [
                _format_price(row.get("price")),
                f"⭐ {float(rating):.1f}" if pd.notna(rating) else None,
                f"{int(reviews):,} reviews" if pd.notna(reviews) else None,
                row.get("category"),
            ]
            st.caption(" · ".join(str(b) for b in bits if b))

            if show_reason:
                reasons = explain.explain_row(row)
                st.markdown(
                    " ".join(
                        f":{_reason_colour(r)}-badge[{r}]" for r in reasons
                    )
                )


def _render_recommendation(row: pd.Series) -> None:
    with st.container(border=True):
        if _has_image(row.get("image_url")):
            st.image(row["image_url"], use_container_width=True)
        title = str(row.get("title", ""))[:55]
        url = row.get("product_url")
        st.caption(f"[{title}]({url})" if _has_image(url) else title)
        rating = row.get("rating")
        star = f" · ⭐ {float(rating):.1f}" if pd.notna(rating) else ""
        st.markdown(f"**{_format_price(row.get('price'))}**{star}")


def main() -> None:
    st.set_page_config(
        page_title="Search The Product",
        page_icon="🔎",
        layout="centered",
    )

    st.title("🔎 Search The Product")
    st.caption(
        "Browse **728 men's clothing & footwear products** — jeans, polos, t-shirts, "
        "shirts, sweatpants, shorts, hoodies, jackets, underwear, sneakers and running "
        "shoes. Search in plain English (e.g. *“slim fit jeans for work”* or *“kicks for "
        "the gym”*) and results are ranked by **what you really mean** — blending text "
        "match with rating, price, reviews and popularity — and each one explains *why* "
        "it's shown."
    )

    engine = get_engine()

    if "query" not in st.session_state:
        st.session_state["query"] = "moisture wicking golf polo"

    st.text_input(
        "Search",
        key="query",
        placeholder="e.g. slim fit jeans for work",
        label_visibility="collapsed",
    )

    st.caption("Try one:")
    for col, example in zip(st.columns(len(_EXAMPLES)), _EXAMPLES):
        col.button(
            example,
            use_container_width=True,
            on_click=_set_query,
            args=(example,),
        )

    left, right = st.columns([1, 1], vertical_alignment="center")
    with left:
        top_k = st.slider("How many results?", min_value=3, max_value=20, value=8)
    with right:
        show_compare = st.toggle("Compare with plain keyword search", value=False)

    query = st.session_state["query"]
    if not query.strip():
        st.info("Enter a search above to see ranked products.")
        return

    st.divider()

    results = ranking.rank_products(query, engine, top_k=top_k)

    if show_compare:
        baseline = ranking.rank_by_cosine(query, engine, top_k=top_k)
        ml_col, base_col = st.columns(2)
        with ml_col:
            st.subheader("🤖 Smart ML ranking")
            for i, (_, row) in enumerate(results.iterrows(), start=1):
                _render_product(row, i, show_reason=True)
        with base_col:
            st.subheader("🔤 Keyword-only (baseline)")
            for i, (_, row) in enumerate(baseline.iterrows(), start=1):
                _render_product(row, i, show_reason=False)
    else:
        st.subheader(f"Top {len(results)} results")
        for i, (_, row) in enumerate(results.iterrows(), start=1):
            _render_product(row, i, show_reason=True)

    if len(results) > 0:
        st.divider()
        st.subheader("🛍️ You may also like")
        top_asin = results.iloc[0]["asin"]
        recs = recommender.similar_products(
            top_asin, engine.products, engine.product_matrix, top_k=5
        )
        if len(recs):
            for col, (_, row) in zip(st.columns(len(recs)), recs.iterrows()):
                with col:
                    _render_recommendation(row)


if __name__ == "__main__":
    main()
