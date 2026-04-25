FROM python:3.11-slim

WORKDIR /app

# System deps for lxml / wordcloud
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libxml2-dev \
    libxslt1-dev \
    libfreetype6-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK assets
RUN python -c "import nltk; [nltk.download(p, quiet=True) for p in ('stopwords','wordnet','omw-1.4','punkt')]"

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "dashboard/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
