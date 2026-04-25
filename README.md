# 🧠 AI Product Review Sentiment Analyzer

An end-to-end NLP project that **scrapes product reviews from e-commerce sites**, cleans the text, and uses **Machine Learning + BERT** to classify reviews as ⭐ Positive, 🤷 Neutral, or 👎 Negative. Includes an interactive **Streamlit dashboard** with visualisations, word clouds, and review insights.

---

## 📂 Project Structure

```
sentiment-analyzer/
│
├── data/
│   ├── raw_reviews.csv          # Scraped reviews
│   └── cleaned_reviews.csv      # Preprocessed reviews
│
├── scraper/
│   └── scraper.py               # Amazon / Flipkart / BestBuy scraper
│
├── preprocessing/
│   └── preprocess.py            # Text cleaning pipeline
│
├── models/
│   ├── baseline_model.py        # TF-IDF + LR / NB / SVM trainer
│   ├── baseline_model.pkl       # Saved best model (auto-generated)
│   ├── bert_model.py            # DistilBERT fine-tuning & inference
│   └── bert_model/              # Saved BERT weights (auto-generated)
│
├── dashboard/
│   └── app.py                   # Streamlit dashboard
│
├── notebooks/
│   ├── EDA.ipynb                # Exploratory Data Analysis
│   └── Model_Training.ipynb     # Model training walkthrough
│
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate sample data (or scrape real reviews)

```bash
# Demo mode (generates 200 synthetic reviews)
python scraper/scraper.py --site demo

# Real scraping (Amazon example)
python scraper/scraper.py --url "https://www.amazon.in/dp/B0XXXXXXXX" --site amazon --pages 5
```

### 3. Preprocess

```bash
python preprocessing/preprocess.py
```

### 4. Train baseline models

```bash
python models/baseline_model.py --all
```

### 5. (Optional) Fine-tune BERT

```bash
# Recommended on GPU / Google Colab
python models/bert_model.py --train
```

### 6. Launch the dashboard

```bash
streamlit run dashboard/app.py
```

---

## 🧰 Tech Stack

| Layer | Tools |
|-------|-------|
| Scraping | `requests`, `BeautifulSoup`, `Selenium` |
| Preprocessing | `NLTK`, `re`, `pandas` |
| ML Models | `scikit-learn` (TF-IDF, LR, NB, SVM) |
| Deep Learning | `HuggingFace Transformers`, `PyTorch` (DistilBERT) |
| Visualisation | `Plotly`, `Matplotlib`, `WordCloud` |
| Dashboard | `Streamlit` |
| Deployment | Docker, Streamlit Cloud, HuggingFace Spaces |

---

## 🐳 Docker

```bash
docker build -t sentiment-analyzer .
docker run -p 8501:8501 sentiment-analyzer
# Open http://localhost:8501
```

---

## 📊 Dashboard Pages

| Page | Description |
|------|-------------|
| 📊 Dashboard | KPI cards, sentiment charts, word clouds, review table |
| 🔍 Live Prediction | Enter any review → instant ML or BERT prediction |
| 🌐 Scrape & Analyze | Paste a product URL → scrape + analyse on the fly |
| ℹ️ About | Project overview and pipeline description |

---

## 🧪 Future Improvements

- Multi-product comparison
- Topic modelling (LDA / BERTopic)
- Automatic review summarisation
- Browser extension integration
- PostgreSQL storage for scraped products
