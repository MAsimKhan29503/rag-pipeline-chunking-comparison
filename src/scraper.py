"""
Step 1: Scrape (Bronze layer)

Pulls raw HTML from each URL in config.SOURCE_URLS, strips it down to the
main readable text, and saves one .txt file per page under data/raw/.

Run this with:
    python src/scraper.py
"""
import os
import re
import time
import requests
from bs4 import BeautifulSoup

import config


def slug_from_url(url: str) -> str:
    """Turn a URL into a safe filename, e.g. sql-programming-guide.txt"""
    name = url.rstrip("/").split("/")[-1]
    name = name.replace(".html", "") or "index"
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name) + ".txt"


def fetch_page(url: str, timeout: int = 15) -> str:
    """Fetch a page and return its raw HTML. Raises on HTTP errors."""
    headers = {
        "User-Agent": "RAG-portfolio-project/1.0 (educational use)"
    }
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def html_to_clean_text(html: str) -> str:
    """
    Strip navigation, scripts, styles, and boilerplate; keep the main
    readable content.
    """
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()

    main = (
        soup.find("div", {"class": "content"})
        or soup.find("div", {"id": "content"})
        or soup.find("article")
        or soup.body
    )
    if main is None:
        main = soup

    text = main.get_text(separator="\n")

    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def scrape_all(urls=None, out_dir=None, delay_seconds: float = 1.5):
    """
    Scrape every URL, save cleaned text to out_dir, and return a list of
    (url, filepath) pairs for pages that succeeded.
    """
    urls = urls or config.SOURCE_URLS
    out_dir = out_dir or config.RAW_DIR
    os.makedirs(out_dir, exist_ok=True)

    results = []
    for i, url in enumerate(urls):
        print(f"[{i+1}/{len(urls)}] Fetching {url}")
        try:
            html = fetch_page(url)
            text = html_to_clean_text(html)
            filename = slug_from_url(url)
            filepath = os.path.join(out_dir, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"    -> saved {len(text)} chars to {filepath}")
            results.append((url, filepath))
        except requests.RequestException as e:
            print(f"    !! failed to fetch {url}: {e}")

        if i < len(urls) - 1:
            time.sleep(delay_seconds)

    return results


if __name__ == "__main__":
    scraped = scrape_all()
    print(f"\nDone. Successfully scraped {len(scraped)}/{len(config.SOURCE_URLS)} pages.")