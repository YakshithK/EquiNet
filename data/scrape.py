# blog_crawler.py

import os
import json
from newspaper import Article
from datetime import datetime
from langdetect import detect
import nltk 
nltk.download('punkt_tab')

# -----------------------------
# CONFIGURATION
# -----------------------------
BLOG_URLS = [
    "https://globalvoices.org/",
    "https://globalvoices.org/category/climate/",
    "https://www2.gov.bc.ca/gov/content/environment/natural-resource-stewardship/consulting-with-first-nations/first-nations-clean-energy-business-fund",
    "https://www.aqam.net/operations/%CA%94aqant%E2%B1%A1a%CA%94nam-housing/",
    "https://ecotrust.ca/latest/blog/energy-security-work-continues-in-three-first-nation-communities/",
    "https://climateatlas.ca/agriculture-and-climate-change",
    "https://climateatlas.ca/long-sun-shines",
    "https://climateatlas.ca/basics-climate-risk-assessment",
    "https://climateatlas.ca/building-climate-resilient-city",
    "https://climateatlas.ca/canadian-cities-and-climate-change",
    "https://climateatlas.ca/climate-atlas-v2-why-have-numbers-changed",
    "https://climateatlas.ca/climate-change-and-health",
    "https://climateatlas.ca/climate-change-maps-agriculture",
    "https://climateatlas.ca/climate-change-projections",
    "https://climateatlas.ca/climate-change-air-quality-and-public-health",
    "https://climateatlas.ca/climate-change-basics",
    "https://climateatlas.ca/climate-maps-cities",
    "https://climateatlas.ca/indigenous-knowledges-and-climate-change",
    "https://climateatlas.ca/indigenous/first-nations",
    "https://climateatlas.ca/indigenous/inuit",
]

OUTPUT_FILE = "blogs_dataset.json"
MAX_ARTICLES_PER_SITE = 20  # limit to speed up crawling for hackathon

# -----------------------------
# FUNCTIONS
# -----------------------------
def crawl_article(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        article.nlp()

        return {
            "url": url,
            "title": article.title,
            "text": article.text,
            "authors": article.authors,
            "publish_date": str(article.publish_date) if article.publish_date else None,
            "keywords": article.keywords,
            "summary": article.summary,
            "language": detect(article.text) if article.text else None,
            "source": url.split("/")[2]  # domain name
        }

    except Exception as e:
        print(f"[ERROR] Failed to crawl {url}: {e}")
        return None


def crawl_site(url, max_articles=20):
    from newspaper import build
    print(f"[INFO] Crawling site: {url}")
    site = build(url, memoize_articles=False)
    articles_data = []

    for article in site.articles[:max_articles]:
        print(f"[INFO] Crawling article: {article.url}")
        data = crawl_article(article.url)
        if data:
            articles_data.append(data)

    return articles_data


# -----------------------------
# MAIN SCRIPT
# -----------------------------
if __name__ == "__main__":
    all_articles = []

    for blog_url in BLOG_URLS:
        articles = crawl_site(blog_url, MAX_ARTICLES_PER_SITE)
        all_articles.extend(articles)

    print(f"[INFO] Crawled {len(all_articles)} articles in total.")

    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
    else:
        existing_data = []

    existing_data.extend(all_articles)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Saved data to {OUTPUT_FILE}")
