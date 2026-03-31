import httpx
import os
import json
from datetime import date
from urllib.parse import urlparse

SECTION_URL = "https://www.ecfr.gov/current/title-34/subtitle-B/chapter-VI/part-668/section-668.32"

SAVE_DIR = "data/raw_html/ecfr"
os.makedirs(SAVE_DIR, exist_ok=True)
RAW_HTML_PATH = f"{SAVE_DIR}/section_668_32.html"
RAW_META_PATH = f"{SAVE_DIR}/section_668_32_source.json"
BLOCKED_HTML_PATH = f"{SAVE_DIR}/section_668_32_blocked.html"

BLOCK_MARKERS = [
    "aggressive automated scraping",
    "complete the captcha",
    "programmatic access to these sites is limited",
    "your request has been flagged as potentially automated",
]

EXPECTED_CONTENT_MARKERS = [
    "p-668.32(",
    "§ 668.32",
    "student eligibility",
]


def build_source_name(url: str) -> str:
    parsed = urlparse(url)
    host = (parsed.netloc or "source").replace("www.", "")
    tail = parsed.path.strip("/").split("/")[-1] if parsed.path else "page"
    return f"{host}::{tail}"


async def fetch_page(url):
    async with httpx.AsyncClient(follow_redirects=True) as client:
        # Keep this request minimal to match the simple scraper behavior
        # that succeeds for this endpoint.
        res = await client.get(url)
        res.raise_for_status()
        return res.text


def is_blocked_html(html: str) -> bool:
    low = (html or "").lower()
    # If expected section markers exist, treat as valid content.
    if any(marker in low for marker in EXPECTED_CONTENT_MARKERS):
        return False

    matches = sum(1 for marker in BLOCK_MARKERS if marker in low)
    return matches >= 2


async def fetch_ecfr_html():
    html = await fetch_page(SECTION_URL)
    if is_blocked_html(html):
        with open(BLOCKED_HTML_PATH, "w", encoding="utf-8") as f:
            f.write(html)
        raise RuntimeError(
            "Blocked by anti-bot/CAPTCHA page from eCFR. "
            f"Saved blocked response to: {BLOCKED_HTML_PATH}"
        )

    with open(RAW_HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html)

    source_meta = {
        "source_name": build_source_name(SECTION_URL),
        "source_url": SECTION_URL,
        "source_type": "cfr",
        "authority_level": "regulation",
        "document_title": "34 CFR §668.32",
        "section_ref": "§668.32",
        "effective_date": "",
        "ingestion_date": date.today().isoformat(),
        "volume": "",
        "chapter": "",
        "title": "34",
        "part": "668",
    }
    with open(RAW_META_PATH, "w", encoding="utf-8") as f:
        json.dump(source_meta, f, indent=2, ensure_ascii=False)

    print(f"Saved raw HTML to: {RAW_HTML_PATH}")
    print(f"Saved source metadata to: {RAW_META_PATH}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(fetch_ecfr_html())