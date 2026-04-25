"""
scraper.py — Scrape product reviews from Amazon / BestBuy / Flipkart.

Usage:
    python scraper.py --url "https://www.amazon.in/dp/ASIN" --pages 5 --site amazon
"""

import argparse
import csv
import os
import random
import time
from datetime import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw_reviews.csv")


# ──────────────────────────────────────────────────────────────────────────────
# Amazon scraper
# ──────────────────────────────────────────────────────────────────────────────

def scrape_amazon(url: str, max_pages: int = 5) -> list[dict]:
    """Scrape Amazon product reviews."""
    reviews = []
    # Extract ASIN from URL
    asin = _extract_amazon_asin(url)
    base_url = f"https://www.amazon.in/product-reviews/{asin}/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews"

    for page in range(1, max_pages + 1):
        page_url = f"{base_url}&pageNumber={page}"
        print(f"[Amazon] Scraping page {page}: {page_url}")
        soup = _get_soup(page_url)
        if soup is None:
            break

        review_divs = soup.find_all("div", {"data-hook": "review"})
        if not review_divs:
            print(f"[Amazon] No reviews found on page {page}. Stopping.")
            break

        for div in review_divs:
            try:
                title = div.find("a", {"data-hook": "review-title"})
                body = div.find("span", {"data-hook": "review-body"})
                rating_elem = div.find("i", {"data-hook": "review-star-rating"})
                date_elem = div.find("span", {"data-hook": "review-date"})
                reviewer_elem = div.find("span", class_="a-profile-name")

                reviews.append({
                    "title": title.get_text(strip=True) if title else "",
                    "review": body.get_text(strip=True) if body else "",
                    "rating": _parse_amazon_rating(rating_elem),
                    "date": date_elem.get_text(strip=True) if date_elem else "",
                    "reviewer": reviewer_elem.get_text(strip=True) if reviewer_elem else "Anonymous",
                    "source": "amazon",
                    "scraped_at": datetime.now().isoformat(),
                })
            except Exception as e:
                print(f"[Amazon] Error parsing review: {e}")

        time.sleep(random.uniform(1.5, 3.5))

    return reviews


def _extract_amazon_asin(url: str) -> str:
    import re
    match = re.search(r"/dp/([A-Z0-9]{10})", url)
    if match:
        return match.group(1)
    raise ValueError(f"Could not extract ASIN from URL: {url}")


def _parse_amazon_rating(elem) -> float:
    if elem is None:
        return 0.0
    text = elem.get_text(strip=True)  # e.g. "4.0 out of 5 stars"
    try:
        return float(text.split()[0])
    except (ValueError, IndexError):
        return 0.0


# ──────────────────────────────────────────────────────────────────────────────
# BestBuy scraper
# ──────────────────────────────────────────────────────────────────────────────

def scrape_bestbuy(url: str, max_pages: int = 5) -> list[dict]:
    """Scrape BestBuy product reviews."""
    reviews = []
    # Extract SKU from BestBuy URL
    import re
    sku_match = re.search(r"/(\d+)\.p", url)
    if not sku_match:
        raise ValueError(f"Could not extract SKU from BestBuy URL: {url}")
    sku = sku_match.group(1)

    api_base = (
        f"https://www.bestbuy.com/api/3.0/prj-bfy/api/2.0/"
        f"reviews/?apiKey=D244FDA50D9EA838E8C5CBB69B7D5D34&sku={sku}"
        f"&sort=BEST_MATCH&page={{page}}&pageSize=20"
    )

    for page in range(1, max_pages + 1):
        print(f"[BestBuy] Scraping page {page}")
        resp = _safe_get(api_base.format(page=page))
        if resp is None:
            break
        data = resp.json()
        items = data.get("topics", [])
        if not items:
            break

        for item in items:
            reviews.append({
                "title": item.get("title", ""),
                "review": item.get("text", ""),
                "rating": item.get("rating", 0),
                "date": item.get("createdDate", ""),
                "reviewer": item.get("userNickname", "Anonymous"),
                "source": "bestbuy",
                "scraped_at": datetime.now().isoformat(),
            })
        time.sleep(random.uniform(1.0, 2.5))

    return reviews


# ──────────────────────────────────────────────────────────────────────────────
# Flipkart scraper  (HTML-based, no JS rendering needed for basic content)
# ──────────────────────────────────────────────────────────────────────────────

def scrape_flipkart(url: str, max_pages: int = 5) -> list[dict]:
    """Scrape Flipkart product reviews."""
    reviews = []
    # Build paginated review URL
    base = url.split("?")[0]
    params_base = "marketplace=FLIPKART&page={page}"

    for page in range(1, max_pages + 1):
        page_url = f"{base}?{params_base.format(page=page)}"
        print(f"[Flipkart] Scraping page {page}: {page_url}")
        soup = _get_soup(page_url)
        if soup is None:
            break

        review_blocks = soup.find_all("div", class_="col EPCmJX Ma-hfCU")
        if not review_blocks:
            # Alternate class names change over time
            review_blocks = soup.find_all("div", class_=lambda c: c and "review" in c.lower())
        if not review_blocks:
            print(f"[Flipkart] No reviews found on page {page}.")
            break

        for block in review_blocks:
            try:
                title = block.find("p", class_=lambda c: c and "z9E0IG" in (c or ""))
                body = block.find("div", class_=lambda c: c and "ZmyHeo" in (c or ""))
                rating_elem = block.find("div", class_=lambda c: c and "XQDdHH" in (c or ""))
                reviewer_elem = block.find("p", class_=lambda c: c and "MDs1cf" in (c or ""))

                reviews.append({
                    "title": title.get_text(strip=True) if title else "",
                    "review": body.get_text(strip=True) if body else "",
                    "rating": float(rating_elem.get_text(strip=True)) if rating_elem else 0.0,
                    "date": "",
                    "reviewer": reviewer_elem.get_text(strip=True) if reviewer_elem else "Anonymous",
                    "source": "flipkart",
                    "scraped_at": datetime.now().isoformat(),
                })
            except Exception as e:
                print(f"[Flipkart] Error parsing review: {e}")

        time.sleep(random.uniform(1.5, 3.0))

    return reviews


# ──────────────────────────────────────────────────────────────────────────────
# Demo / sample data generator (when real scraping is blocked)
# ──────────────────────────────────────────────────────────────────────────────

def generate_sample_reviews(n: int = 200) -> list[dict]:
    """Generate realistic synthetic reviews for demo purposes."""
    import random as rnd
    positives = [
        "Absolutely love this product! Best purchase I've made this year.",
        "Excellent quality. Arrived on time and works perfectly.",
        "Five stars without hesitation. Highly recommend to everyone.",
        "Great value for money. Very satisfied with the quality.",
        "Works exactly as described. Really happy with this purchase.",
        "Superb build quality and fast delivery. Will buy again.",
        "Outstanding product! Exceeded all my expectations.",
        "Very impressed with the quality and performance.",
    ]
    neutrals = [
        "It's okay. Does what it's supposed to, nothing special.",
        "Average product. Not bad, not great.",
        "Decent quality for the price. Some minor issues but manageable.",
        "Works fine. Delivery was a bit slow.",
        "Product is okay. Instructions could be clearer.",
        "It's alright. Could be better, could be worse.",
    ]
    negatives = [
        "Very disappointed. Product stopped working after a week.",
        "Poor quality. Not worth the money at all.",
        "Terrible experience. Customer service was unhelpful.",
        "Broke after first use. Complete waste of money.",
        "Do not buy this. Quality is extremely poor.",
        "Worst product I've ever bought. Returning immediately.",
        "Misleading description. What arrived was nothing like the photos.",
    ]

    reviews = []
    for i in range(n):
        sentiment = rnd.choices(["positive", "neutral", "negative"], weights=[0.6, 0.2, 0.2])[0]
        if sentiment == "positive":
            text = rnd.choice(positives)
            rating = rnd.uniform(4.0, 5.0)
        elif sentiment == "neutral":
            text = rnd.choice(neutrals)
            rating = rnd.uniform(2.5, 3.5)
        else:
            text = rnd.choice(negatives)
            rating = rnd.uniform(1.0, 2.0)

        reviews.append({
            "title": text[:40],
            "review": text,
            "rating": round(rating, 1),
            "date": f"2024-{rnd.randint(1,12):02d}-{rnd.randint(1,28):02d}",
            "reviewer": f"User_{rnd.randint(1000,9999)}",
            "source": rnd.choice(["amazon", "flipkart", "bestbuy"]),
            "scraped_at": datetime.now().isoformat(),
        })
    return reviews


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _get_soup(url: str) -> BeautifulSoup | None:
    resp = _safe_get(url)
    if resp is None:
        return None
    return BeautifulSoup(resp.content, "html.parser")


def _safe_get(url: str, retries: int = 3):
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code == 200:
                return resp
            print(f"[HTTP {resp.status_code}] {url}")
        except requests.RequestException as e:
            print(f"[Attempt {attempt+1}] Request error: {e}")
        time.sleep(2 ** attempt)
    return None


def save_reviews(reviews: list[dict], path: str = OUTPUT_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(reviews)
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} reviews → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

SCRAPERS = {
    "amazon": scrape_amazon,
    "bestbuy": scrape_bestbuy,
    "flipkart": scrape_flipkart,
    "demo": lambda url, pages: generate_sample_reviews(200),
}


def main():
    parser = argparse.ArgumentParser(description="Scrape product reviews")
    parser.add_argument("--url", default="demo", help="Product page URL (use 'demo' for sample data)")
    parser.add_argument("--site", choices=list(SCRAPERS.keys()), default="demo")
    parser.add_argument("--pages", type=int, default=5, help="Number of pages to scrape")
    parser.add_argument("--out", default=OUTPUT_PATH, help="Output CSV path")
    args = parser.parse_args()

    scraper_fn = SCRAPERS[args.site]
    reviews = scraper_fn(args.url, args.pages)
    if reviews:
        save_reviews(reviews, args.out)
    else:
        print("No reviews scraped.")


if __name__ == "__main__":
    main()
