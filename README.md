# Intent-Aware Search + Ranking System

A mini e-commerce search engine that understands what a shopper *means*, not just the
keywords they type. Search `"moisture wicking golf polo"` and it returns a **ranked** list
of products, tells you **why** each one ranked where it did, and suggests **similar items**
("you may also like").

Built end to end on **real scraped Amazon data** (728 men's apparel & footwear products,
6,327 reviews) using classic, explainable ML — no black boxes.
Live: https://intent-aware-search-ranking-system.onrender.com

---

## What it does

| Capability | How |
|---|---|
| **Intent-aware search** | TF-IDF text matching + a trained ranker, not plain keyword overlap |
| **Smart ranking** | A Random Forest blends 6 signals (text match, rating, price, reviews, sentiment, popularity) |
| **Recommendations** | Product-to-product cosine similarity ("you may also like") |
| **Explainability** | Plain-English reasons: *"Strong match · Highly rated (4.8/5) · Budget-friendly price"* |

---

## How it works

```
query ──▶ TF-IDF ──▶ intent guard ──▶ 6 features per product ──▶ ML ranker ──▶ sorted results
                                                                    │
                                                                    ├──▶ "why ranked?" explanations
                                                                    └──▶ "you may also like" recommendations
```

The training labels use a **multiplicative gate** — the text match decides *whether* a
product is relevant at all, and quality signals (rating, reviews, price, popularity) decide
the *order* among relevant products. This keeps results on-topic **and** surfaces the best
items first.

---

## Quick start

```powershell
# install dependencies (or use the provided .venv)
pip install -r requirements.txt

# build the data + model (run in order)
python -m src.preprocessing   # raw CSVs  -> data/cleaned_products.csv
python -m src.features        # cleaned   -> TF-IDF artifacts in models/
python -m src.model           # features  -> training data + models/ranker.pkl

# (optional) score the system
python -m src.evaluate

# launch the web app
streamlit run app.py
```


