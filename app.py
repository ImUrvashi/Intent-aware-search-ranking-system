from __future__ import annotations

import html

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

_CSS = """
<style>
.block-container { padding-top: 2rem; max-width: 1200px; }

.hero {
    background: linear-gradient(135deg, #4f46e5 0%, #6366f1 60%, #818cf8 100%);
    border-radius: 18px;
    padding: 28px 32px;
    color: #ffffff;
    margin-bottom: 6px;
}
.hero h1 { margin: 0 0 6px 0; font-size: 1.9rem; }
.hero p { margin: 0; opacity: 0.92; font-size: 0.95rem; line-height: 1.5; }

.section-label {
    font-weight: 700;
    font-size: 0.8rem;
    letter-spacing: .04em;
    text-transform: uppercase;
    color: #64748b;
    margin: 18px 0 8px 0;
}

.product-grid, .rec-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(230px, 1fr));
    gap: 16px;
    margin-top: 4px;
}
.rec-grid { grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); }

.product-card, .rec-card {
    position: relative;
    background: #ffffff;
    border: 1px solid #eef0f4;
    border-radius: 14px;
    padding: 14px;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
    transition: transform .15s ease, box-shadow .15s ease;
    display: flex;
    flex-direction: column;
}
.product-card:hover, .rec-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 24px rgba(15, 23, 42, 0.12);
}

.card-rank {
    position: absolute;
    top: 10px;
    left: 10px;
    background: #4f46e5;
    color: #ffffff;
    font-size: 11px;
    font-weight: 700;
    padding: 2px 9px;
    border-radius: 999px;
    z-index: 2;
}

.card-img-wrap, .rec-img-wrap {
    width: 100%;
    height: 150px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #f8fafc;
    border-radius: 10px;
    overflow: hidden;
    margin-bottom: 10px;
}
.rec-img-wrap { height: 110px; }
.card-img-wrap img, .rec-img-wrap img { max-width: 100%; max-height: 100%; object-fit: contain; }
.no-img { color: #cbd5e1; font-size: 12px; }

.card-title, .rec-title {
    font-size: 13.5px;
    font-weight: 600;
    color: #0f172a;
    text-decoration: none;
    line-height: 1.3;
    display: block;
    margin-bottom: 6px;
}
.card-title:hover, .rec-title:hover { color: #4f46e5; }

.card-meta, .rec-meta {
    font-size: 12px;
    color: #64748b;
    margin-bottom: 8px;
}

.badge-row { display: flex; flex-wrap: wrap; gap: 6px; margin-top: auto; }
.badge {
    font-size: 10.5px;
    font-weight: 600;
    padding: 3px 9px;
    border-radius: 999px;
    white-space: nowrap;
}
.badge-green  { background: #dcfce7; color: #166534; }
.badge-orange { background: #ffedd5; color: #9a3412; }
.badge-blue   { background: #dbeafe; color: #1e40af; }
.badge-violet { background: #ede9fe; color: #5b21b6; }
.badge-red    { background: #fee2e2; color: #991b1b; }
.badge-gray   { background: #f1f5f9; color: #334155; }

.filter-note { font-size: 12.5px; color: #94a3b8; margin-top: 4px; }
</style>
"""


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


def _apply_filters(
    df: pd.DataFrame,
    categories: list[str],
    price_range: tuple[float, float],
    min_rating: float,
) -> pd.DataFrame:
    out = df
    if categories:
        out = out[out["category"].isin(categories)]
    if "price" in out.columns:
        low, high = price_range
        out = out[out["price"].isna() | out["price"].between(low, high)]
    if min_rating > 0 and "rating" in out.columns:
        out = out[out["rating"].fillna(0) >= min_rating]
    return out


def _product_card_html(row: pd.Series, rank: int, *, show_reason: bool) -> str:
    title = html.escape(str(row.get("title", "Untitled product")))
    url = row.get("product_url")
    img = row.get("image_url")

    meta_bits = [_format_price(row.get("price"))]
    rating = row.get("rating")
    if pd.notna(rating):
        meta_bits.append(f"⭐ {float(rating):.1f}")
    reviews = row.get("review_count")
    if pd.notna(reviews):
        meta_bits.append(f"{int(reviews):,} reviews")
    category = row.get("category")
    if category:
        meta_bits.append(html.escape(str(category)))
    meta = " · ".join(meta_bits)

    img_html = (
        f'<img src="{html.escape(str(img))}" loading="lazy" alt="">'
        if _has_image(img) else '<div class="no-img">No image</div>'
    )
    title_html = (
        f'<a class="card-title" href="{html.escape(str(url))}" target="_blank" rel="noopener">{title}</a>'
        if _has_image(url) else f'<span class="card-title">{title}</span>'
    )

    badges_html = ""
    if show_reason:
        reasons = explain.explain_row(row)
        spans = "".join(
            f'<span class="badge badge-{_reason_colour(r)}">{html.escape(r)}</span>'
            for r in reasons
        )
        badges_html = f'<div class="badge-row">{spans}</div>'

    return (
        f'<div class="product-card">'
        f'<div class="card-rank">#{rank}</div>'
        f'<div class="card-img-wrap">{img_html}</div>'
        f'{title_html}'
        f'<div class="card-meta">{meta}</div>'
        f'{badges_html}'
        f'</div>'
    )


def _rec_card_html(row: pd.Series) -> str:
    title = html.escape(str(row.get("title", ""))[:60])
    url = row.get("product_url")
    img = row.get("image_url")
    rating = row.get("rating")

    img_html = (
        f'<img src="{html.escape(str(img))}" loading="lazy" alt="">'
        if _has_image(img) else '<div class="no-img">No image</div>'
    )
    title_html = (
        f'<a class="rec-title" href="{html.escape(str(url))}" target="_blank" rel="noopener">{title}</a>'
        if _has_image(url) else f'<span class="rec-title">{title}</span>'
    )
    star = f" · ⭐ {float(rating):.1f}" if pd.notna(rating) else ""

    return (
        f'<div class="rec-card">'
        f'<div class="rec-img-wrap">{img_html}</div>'
        f'{title_html}'
        f'<div class="rec-meta">{_format_price(row.get("price"))}{star}</div>'
        f'</div>'
    )


def _render_product_grid(results: pd.DataFrame, *, show_reason: bool) -> None:
    cards = "".join(
        _product_card_html(row, i, show_reason=show_reason)
        for i, (_, row) in enumerate(results.iterrows(), start=1)
    )
    st.markdown(f'<div class="product-grid">{cards}</div>', unsafe_allow_html=True)


def _render_sidebar_filters(products: pd.DataFrame):
    st.sidebar.header("🔧 Filters")

    categories = sorted(c for c in products["category"].dropna().unique())
    selected_categories = st.sidebar.multiselect("Category", categories, default=[])

    price_min = float(products["price"].min())
    price_max = float(products["price"].max())
    price_range = st.sidebar.slider(
        "Price range ($)",
        min_value=price_min,
        max_value=price_max,
        value=(price_min, price_max),
    )

    min_rating = st.sidebar.slider("Minimum rating", 0.0, 5.0, 0.0, 0.5)

    st.sidebar.caption(
        "Filters apply to the ranked results below — they narrow, not replace, the search."
    )
    return selected_categories, price_range, min_rating


def main() -> None:
    st.set_page_config(
        page_title="Search The Product",
        page_icon="🔎",
        layout="wide",
    )
    st.markdown(_CSS, unsafe_allow_html=True)

    engine = get_engine()

    st.markdown(
        '<div class="hero"><h1>🔎 Search The Product</h1>'
        "<p>Browse <b>728 men's clothing & footwear products</b> — jeans, polos, t-shirts, "
        "shirts, sweatpants, shorts, hoodies, jackets, underwear, sneakers and running shoes. "
        "Search in plain English (e.g. “slim fit jeans for work” or “kicks for the gym”) and "
        "results are ranked by what you really mean — blending text match with rating, price, "
        "reviews and popularity — and each one explains why it's shown.</p></div>",
        unsafe_allow_html=True,
    )

    selected_categories, price_range, min_rating = _render_sidebar_filters(engine.products)

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

    # Rank a larger candidate pool than requested so filters have room to work with.
    candidate_k = min(len(engine.products), max(top_k * 5, 50))

    def _ranked_and_filtered(rank_fn) -> tuple[pd.DataFrame, int]:
        candidates = rank_fn(query, engine, top_k=candidate_k)
        filtered = _apply_filters(candidates, selected_categories, price_range, min_rating)
        return filtered.head(top_k), len(filtered)

    results, n_matching = _ranked_and_filtered(ranking.rank_products)

    st.markdown('<div class="section-label">Results</div>', unsafe_allow_html=True)

    if show_compare:
        baseline, _ = _ranked_and_filtered(ranking.rank_by_cosine)
        ml_col, base_col = st.columns(2)
        with ml_col:
            st.subheader("🤖 Smart ML ranking")
            if len(results):
                _render_product_grid(results, show_reason=True)
            else:
                st.info("No products match the current filters.")
        with base_col:
            st.subheader("🔤 Keyword-only (baseline)")
            if len(baseline):
                _render_product_grid(baseline, show_reason=False)
            else:
                st.info("No products match the current filters.")
    else:
        st.subheader(f"Top {len(results)} results")
        if n_matching > len(results):
            st.markdown(
                f'<div class="filter-note">Showing {len(results)} of {n_matching} '
                "matches within your filters — raise the count above to see more.</div>",
                unsafe_allow_html=True,
            )
        if len(results):
            _render_product_grid(results, show_reason=True)
        else:
            st.info("No products match the current filters — try widening the price range or rating.")

    if len(results) > 0:
        st.markdown('<div class="section-label">You may also like</div>', unsafe_allow_html=True)
        top_asin = results.iloc[0]["asin"]
        recs = recommender.similar_products(
            top_asin, engine.products, engine.product_matrix, top_k=5
        )
        if len(recs):
            cards = "".join(_rec_card_html(row) for _, row in recs.iterrows())
            st.markdown(f'<div class="rec-grid">{cards}</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
