"""
app.py — Streamlit dashboard for the AI Review Sentiment Analyzer.

Run:
    streamlit run dashboard/app.py
"""

import os
import sys

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ── Path constants ─────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_CSV = os.path.join(DATA_DIR, "raw_reviews.csv")
CLEAN_CSV = os.path.join(DATA_DIR, "cleaned_reviews.csv")
MODEL_PKL = os.path.join(os.path.dirname(__file__), "..", "models", "baseline_model.pkl")


# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Review Sentiment Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .main { background-color: #f0f4f8; }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center;
    }
    .sentiment-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_data(path: str) -> pd.DataFrame | None:
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


@st.cache_resource
def load_model():
    import pickle
    if os.path.exists(MODEL_PKL):
        with open(MODEL_PKL, "rb") as f:
            return pickle.load(f)
    return None


def generate_sample_data(n: int = 300) -> pd.DataFrame:
    """Generate demo data when no CSV is present."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from scraper.scraper import generate_sample_reviews
    from preprocessing.preprocess import preprocess_dataframe

    raw = pd.DataFrame(generate_sample_reviews(n))
    return preprocess_dataframe(raw)


def predict_sentiment_ml(text: str, model) -> dict:
    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    label = int(model.predict([text])[0])
    proba = None
    if hasattr(model.named_steps["clf"], "predict_proba"):
        proba_arr = model.predict_proba([text])[0]
        proba = {label_map[i]: float(p) for i, p in enumerate(proba_arr)}
    return {"sentiment": label_map[label], "probabilities": proba}


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ──────────────────────────────────────────────────────────────────────────────

SENTIMENT_COLORS = {
    "positive": "#22c55e",
    "neutral": "#f59e0b",
    "negative": "#ef4444",
}
EMOJI = {"positive": "⭐", "neutral": "🤷", "negative": "👎"}


def plot_sentiment_distribution(df: pd.DataFrame):
    counts = df["sentiment"].value_counts().reset_index()
    counts.columns = ["sentiment", "count"]
    fig = px.pie(
        counts,
        names="sentiment",
        values="count",
        color="sentiment",
        color_discrete_map=SENTIMENT_COLORS,
        title="Sentiment Distribution",
        hole=0.45,
    )
    fig.update_traces(textinfo="percent+label")
    fig.update_layout(showlegend=False)
    return fig


def plot_rating_histogram(df: pd.DataFrame):
    fig = px.histogram(
        df,
        x="rating",
        nbins=10,
        color_discrete_sequence=["#6366f1"],
        title="Rating Distribution",
        labels={"rating": "Star Rating", "count": "Count"},
    )
    fig.update_layout(bargap=0.1)
    return fig


def plot_sentiment_over_time(df: pd.DataFrame):
    if "date" not in df.columns or df["date"].isna().all():
        return None
    tmp = df.copy()
    tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")
    tmp = tmp.dropna(subset=["date"])
    if tmp.empty:
        return None
    tmp["month"] = tmp["date"].dt.to_period("M").astype(str)
    grouped = (
        tmp.groupby(["month", "sentiment"])
        .size()
        .reset_index(name="count")
    )
    fig = px.line(
        grouped,
        x="month",
        y="count",
        color="sentiment",
        color_discrete_map=SENTIMENT_COLORS,
        title="Sentiment Over Time",
        markers=True,
    )
    return fig


def generate_wordcloud(texts: list[str], colormap: str = "Blues") -> plt.Figure:
    combined = " ".join(texts)
    wc = WordCloud(
        width=800,
        height=400,
        background_color="white",
        colormap=colormap,
        max_words=150,
        collocations=False,
    ).generate(combined or "no data")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    plt.tight_layout(pad=0)
    return fig


def source_bar(df: pd.DataFrame):
    if "source" not in df.columns:
        return None
    counts = df["source"].value_counts().reset_index()
    counts.columns = ["source", "count"]
    fig = px.bar(
        counts,
        x="source",
        y="count",
        color="source",
        title="Reviews by Source",
        labels={"source": "Platform", "count": "Reviews"},
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image(
        "https://cdn-icons-png.flaticon.com/512/4712/4712126.png",
        width=80,
    )
    st.title("🧠 Sentiment Analyzer")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["📊 Dashboard", "🔍 Live Prediction", "🌐 Scrape & Analyze", "ℹ️ About"],
    )
    st.markdown("---")

    # Filters (only for Dashboard)
    if page == "📊 Dashboard":
        st.subheader("Filters")
        sentiment_filter = st.multiselect(
            "Sentiment",
            ["positive", "neutral", "negative"],
            default=["positive", "neutral", "negative"],
        )
        if st.sidebar.checkbox("Show only long reviews (>50 chars)"):
            min_chars = 50
        else:
            min_chars = 0


# ──────────────────────────────────────────────────────────────────────────────
# Load / create data
# ──────────────────────────────────────────────────────────────────────────────

df_clean = load_data(CLEAN_CSV)
if df_clean is None:
    st.warning("No cleaned data found. Generating sample data for demo …")
    df_clean = generate_sample_data(300)

model = load_model()


# ──────────────────────────────────────────────────────────────────────────────
# PAGE: Dashboard
# ──────────────────────────────────────────────────────────────────────────────

if page == "📊 Dashboard":
    st.title("📊 Review Sentiment Dashboard")

    # Apply filters
    df_view = df_clean[df_clean["sentiment"].isin(sentiment_filter)]
    if min_chars:
        df_view = df_view[df_view["review"].str.len() >= min_chars]

    # KPI cards
    total = len(df_view)
    pos = (df_view["sentiment"] == "positive").sum()
    neg = (df_view["sentiment"] == "negative").sum()
    neu = (df_view["sentiment"] == "neutral").sum()
    avg_rating = df_view["rating"].mean() if "rating" in df_view.columns else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Reviews", f"{total:,}")
    c2.metric("⭐ Positive", f"{pos:,}", f"{100*pos/max(total,1):.1f}%")
    c3.metric("🤷 Neutral", f"{neu:,}", f"{100*neu/max(total,1):.1f}%")
    c4.metric("👎 Negative", f"{neg:,}", f"{100*neg/max(total,1):.1f}%")
    c5.metric("Avg Rating", f"{avg_rating:.2f} ⭐")

    st.markdown("---")

    # Row 1 charts
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_sentiment_distribution(df_view), use_container_width=True)
    with col2:
        if "rating" in df_view.columns:
            st.plotly_chart(plot_rating_histogram(df_view), use_container_width=True)

    # Row 2: timeline + source
    col3, col4 = st.columns(2)
    with col3:
        timeline = plot_sentiment_over_time(df_view)
        if timeline:
            st.plotly_chart(timeline, use_container_width=True)
        else:
            st.info("No date column found for timeline.")
    with col4:
        src_fig = source_bar(df_view)
        if src_fig:
            st.plotly_chart(src_fig, use_container_width=True)

    # Word clouds
    st.markdown("### ☁️ Word Clouds by Sentiment")
    wc1, wc2, wc3 = st.columns(3)
    for col_wc, sent, cmap in zip(
        [wc1, wc2, wc3],
        ["positive", "neutral", "negative"],
        ["Greens", "YlOrBr", "Reds"],
    ):
        texts = df_view[df_view["sentiment"] == sent]["cleaned_text"].dropna().tolist()
        with col_wc:
            st.markdown(f"**{EMOJI[sent]} {sent.title()}**")
            if texts:
                fig = generate_wordcloud(texts, colormap=cmap)
                st.pyplot(fig)
            else:
                st.info("No data.")

    # Review table
    st.markdown("### 📋 Review Table")
    display_cols = [c for c in ["reviewer", "rating", "sentiment", "review"] if c in df_view.columns]
    st.dataframe(
        df_view[display_cols].head(200),
        use_container_width=True,
        hide_index=True,
    )

    # Download
    csv_bytes = df_view.to_csv(index=False).encode()
    st.download_button("⬇️ Download Filtered Data (CSV)", csv_bytes, "filtered_reviews.csv", "text/csv")


# ──────────────────────────────────────────────────────────────────────────────
# PAGE: Live Prediction
# ──────────────────────────────────────────────────────────────────────────────

elif page == "🔍 Live Prediction":
    st.title("🔍 Live Review Sentiment Prediction")
    st.markdown(
        "Enter any product review below to get an instant sentiment prediction."
    )

    review_input = st.text_area("📝 Paste your review here …", height=180, max_chars=2000)

    col_a, col_b = st.columns([1, 3])
    use_bert = col_a.checkbox("Use BERT (if available)", value=False)
    predict_btn = col_b.button("🔮 Predict Sentiment", type="primary")

    if predict_btn and review_input.strip():
        with st.spinner("Analysing …"):
            result = None

            if use_bert:
                try:
                    from models.bert_model import BERTSentimentPredictor
                    predictor = BERTSentimentPredictor()
                    result = predictor.predict(review_input)
                except Exception as e:
                    st.warning(f"BERT unavailable ({e}). Falling back to ML model.")

            if result is None:
                if model:
                    result = predict_sentiment_ml(review_input, model)
                else:
                    # Lightweight fallback using lexicon
                    result = _simple_lexicon_predict(review_input)

        sent = result["sentiment"]
        color = SENTIMENT_COLORS[sent]
        st.markdown(
            f"""
            <div style='background:{color}22; border-left:5px solid {color}; 
                        border-radius:8px; padding:1rem; margin-top:1rem;'>
                <h2 style='color:{color}; margin:0'>{EMOJI[sent]} {sent.title()}</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if result.get("probabilities"):
            proba = result["probabilities"]
            fig_bar = go.Figure(
                go.Bar(
                    x=list(proba.keys()),
                    y=list(proba.values()),
                    marker_color=[SENTIMENT_COLORS[k] for k in proba],
                    text=[f"{v:.1%}" for v in proba.values()],
                    textposition="auto",
                )
            )
            fig_bar.update_layout(title="Confidence", yaxis_tickformat=".0%", height=300)
            st.plotly_chart(fig_bar, use_container_width=True)

    elif predict_btn:
        st.warning("Please enter a review first.")


def _simple_lexicon_predict(text: str) -> dict:
    """Ultra-simple word-match fallback when no model is trained."""
    positive_words = {"great", "excellent", "amazing", "love", "wonderful", "best", "perfect",
                      "happy", "good", "fantastic", "awesome", "brilliant", "outstanding"}
    negative_words = {"bad", "terrible", "awful", "hate", "worst", "poor", "broken",
                      "disappointed", "useless", "horrible", "trash", "junk", "waste"}
    tokens = set(text.lower().split())
    pos_hits = len(tokens & positive_words)
    neg_hits = len(tokens & negative_words)
    if pos_hits > neg_hits:
        sent, probs = "positive", {"negative": 0.1, "neutral": 0.2, "positive": 0.7}
    elif neg_hits > pos_hits:
        sent, probs = "negative", {"negative": 0.7, "neutral": 0.2, "positive": 0.1}
    else:
        sent, probs = "neutral", {"negative": 0.2, "neutral": 0.6, "positive": 0.2}
    return {"sentiment": sent, "probabilities": probs}


# ──────────────────────────────────────────────────────────────────────────────
# PAGE: Scrape & Analyze
# ──────────────────────────────────────────────────────────────────────────────

elif page == "🌐 Scrape & Analyze":
    st.title("🌐 Scrape & Analyze a Product URL")
    st.info(
        "Enter a product URL from Amazon, Flipkart, or BestBuy. "
        "The scraper will collect reviews and run sentiment analysis in real time. "
        "**Note:** Some sites may block automated scraping. Use Demo mode if blocked."
    )

    url_input = st.text_input("🔗 Product URL", placeholder="https://www.amazon.in/dp/…")
    site = st.selectbox("Site", ["amazon", "flipkart", "bestbuy", "demo"])
    pages = st.slider("Pages to scrape", 1, 10, 3)

    if st.button("🚀 Scrape & Analyze", type="primary"):
        from scraper.scraper import SCRAPERS, save_reviews
        from preprocessing.preprocess import preprocess_dataframe

        with st.spinner("Scraping reviews …"):
            try:
                raw_reviews = SCRAPERS[site](url_input or "demo", pages)
                if not raw_reviews:
                    st.error("No reviews scraped. Try Demo mode.")
                else:
                    raw_df = pd.DataFrame(raw_reviews)
                    save_reviews(raw_reviews, RAW_CSV)

                    with st.spinner("Preprocessing …"):
                        df_result = preprocess_dataframe(raw_df)
                        df_result.to_csv(CLEAN_CSV, index=False)

                    st.success(f"Scraped & processed {len(df_result)} reviews!")
                    st.dataframe(df_result[["reviewer", "rating", "sentiment", "review"]].head(50))

                    col_a2, col_b2 = st.columns(2)
                    with col_a2:
                        st.plotly_chart(plot_sentiment_distribution(df_result), use_container_width=True)
                    with col_b2:
                        if "rating" in df_result.columns:
                            st.plotly_chart(plot_rating_histogram(df_result), use_container_width=True)

            except Exception as e:
                st.error(f"Error: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# PAGE: About
# ──────────────────────────────────────────────────────────────────────────────

elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")
    st.markdown(
        """
        ## AI Product Review Sentiment Analyzer

        An end-to-end NLP pipeline that **scrapes** product reviews from e-commerce
        sites, **cleans** and **preprocesses** the text, and **classifies** reviews as
        ⭐ Positive, 🤷 Neutral, or 👎 Negative using both classical ML and BERT.

        ### Pipeline

        | Step | Component | Description |
        |------|-----------|-------------|
        | 1 | `scraper/scraper.py` | Collect reviews via BeautifulSoup / Selenium |
        | 2 | `preprocessing/preprocess.py` | Clean text, lemmatize, remove stopwords |
        | 3 | `models/baseline_model.py` | TF-IDF + Logistic Regression / NB / SVM |
        | 4 | `models/bert_model.py` | Fine-tuned DistilBERT (HuggingFace Transformers) |
        | 5 | `dashboard/app.py` | Interactive Streamlit dashboard |

        ### Tech Stack

        `Python` · `BeautifulSoup` · `Selenium` · `pandas` · `scikit-learn`  
        `HuggingFace Transformers` · `PyTorch` · `Plotly` · `WordCloud` · `Streamlit`

        ### Deployment
        - **Streamlit Cloud** — push to GitHub, connect, deploy in minutes
        - **HuggingFace Spaces** — free GPU inference
        - **Docker** — see `Dockerfile` in repo root

        ---
        Built with ❤️ · [GitHub](https://github.com/nehas005/ai-review-sentiment-analyzer)
        """
    )
