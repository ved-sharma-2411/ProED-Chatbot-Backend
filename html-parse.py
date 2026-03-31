from bs4 import BeautifulSoup
import os
import json
import re
from datetime import date
from collections import Counter
from urllib.parse import urlparse

SAVE_DIR = "data/raw_html/ecfr"
RAW_HTML_PATH = f"{SAVE_DIR}/section_668_32.html"
RAW_SOURCE_META_PATH = f"{SAVE_DIR}/section_668_32_source.json"
JSON_OUTPUT_PATH = f"{SAVE_DIR}/section_668_32.json"
TEXT_OUTPUT_PATH = f"{SAVE_DIR}/section_668_32.txt"
SECTION_ID_RE = re.compile(r"^p-(?P<section>\d+(?:\.\d+)+)(?P<suffix>(?:\([^)]+\))*)$")
HEADING_SECTION_RE = re.compile(r"§\s*(\d+(?:\.\d+)+)")
BLOCK_MARKERS = [
    "aggressive automated scraping",
    "complete the captcha",
    "request access",
    "programmatic access to these sites is limited",
]


def is_blocked_html(html: str) -> bool:
    low = (html or "").lower()
    return any(marker in low for marker in BLOCK_MARKERS)


def build_source_name(url: str) -> str:
    parsed = urlparse(url)
    host = (parsed.netloc or "source").replace("www.", "")
    tail = parsed.path.strip("/").split("/")[-1] if parsed.path else "page"
    return f"{host}::{tail}"


def clean_text(text):
    text = " ".join(text.split())

    # Fix marker spacing: ( a ) ( 1 ) ( i ) -> (a) (1) (i)
    text = re.sub(r"\(\s*([^)]+?)\s*\)", lambda m: f"({m.group(1).strip()})", text)

    # Remove spaces before punctuation: "part ." -> "part."
    text = re.sub(r"\s+([,;:.])", r"\1", text)

    return text


def parse_id_parts(node_id):
    if not node_id:
        return None, []

    match = SECTION_ID_RE.match(node_id)
    if not match:
        return None, []

    section = match.group("section")
    suffix = match.group("suffix")
    parts = re.findall(r"\(([^)]+)\)", suffix)
    return section, parts


def build_subsection_tree(paragraphs):
    root = {"id": None, "label": None, "text": "", "children": []}
    path_index = {(): root}

    for item in paragraphs:
        node_id = item.get("id")
        text = item.get("text", "")
        _, parts = parse_id_parts(node_id)

        path = ()
        for part in parts:
            parent = path_index[path]
            path = (*path, part)
            if path not in path_index:
                node = {"id": None, "label": part, "text": "", "children": []}
                parent["children"].append(node)
                path_index[path] = node

        target = path_index[path]
        target["id"] = node_id
        target["text"] = text

    return root["children"]


def extract_section_content(html):
    soup = BeautifulSoup(html, "lxml")

    # Remove junk
    for tag in soup(["script", "style", "nav"]):
        tag.decompose()

    heading = ""
    heading_section = None
    for h in soup.find_all(["h1", "h2", "h3", "h4", "h5"]):
        h_text = clean_text(h.get_text(" ", strip=True))
        match = HEADING_SECTION_RE.search(h_text)
        if match:
            heading = h_text
            heading_section = match.group(1)
            break

    id_nodes = []
    section_counter = Counter()
    for node in soup.select('[id^="p-"]'):
        node_id = node.get("id")
        section, _ = parse_id_parts(node_id)
        if section:
            id_nodes.append(node)
            section_counter[section] += 1

    target_section = heading_section
    if not target_section and section_counter:
        target_section = section_counter.most_common(1)[0][0]

    paragraphs = []
    for node in id_nodes:
        node_id = node.get("id")
        section, _ = parse_id_parts(node_id)
        if target_section and section != target_section:
            continue

        text = clean_text(node.get_text(" ", strip=True))
        if text:
            paragraphs.append(
                {
                    "id": node_id,
                    "text": text,
                }
            )

    # Fallback if the page structure changes and IDs are not found
    if not paragraphs:
        article = soup.find("article") or soup
        for p in article.find_all("p"):
            text = clean_text(p.get_text(" ", strip=True))
            if text and "ECFR CONTENT" not in text:
                paragraphs.append({"id": None, "text": text})

    subsections = build_subsection_tree(paragraphs)

    return {
        "section": target_section or "",
        "heading": heading,
        "paragraphs": paragraphs,
        "subsections": subsections,
    }


def parse_saved_html():
    if not os.path.exists(RAW_HTML_PATH):
        raise FileNotFoundError(
            f"Raw HTML file not found: {RAW_HTML_PATH}. Run html-scrap.py first."
        )

    with open(RAW_HTML_PATH, "r", encoding="utf-8") as f:
        html = f.read()

    if is_blocked_html(html):
        raise RuntimeError(
            "Detected anti-bot/CAPTCHA HTML in raw source. "
            "Re-run html-scrap.py after resolving access restrictions."
        )

    data = extract_section_content(html)

    fallback_source_url = "https://www.ecfr.gov/current/title-34/subtitle-B/chapter-VI/part-668/section-668.32"
    section_ref = f"§{data['section']}" if data.get("section") else ""
    source_meta = {
        "source_name": build_source_name(fallback_source_url),
        "source_url": fallback_source_url,
        "source_type": "cfr",
        "authority_level": "regulation",
        "document_title": data.get("heading") or "34 CFR §668.32",
        "section_ref": section_ref or "§668.32",
        "effective_date": "",
        "ingestion_date": date.today().isoformat(),
        "volume": "",
        "chapter": "",
        "title": "34",
        "part": "668",
    }
    if os.path.exists(RAW_SOURCE_META_PATH):
        try:
            with open(RAW_SOURCE_META_PATH, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                source_meta.update({k: v for k, v in loaded.items() if v})
        except Exception:
            pass

    data.update(source_meta)

    with open(JSON_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    with open(TEXT_OUTPUT_PATH, "w", encoding="utf-8") as f:
        if data["heading"]:
            f.write(data["heading"] + "\n\n")
        for item in data["paragraphs"]:
            if item["id"]:
                f.write(f"[{item['id']}]\n")
            f.write(item["text"] + "\n\n")

    print(f"Saved parsed JSON to: {JSON_OUTPUT_PATH}")
    print(f"Saved parsed text to: {TEXT_OUTPUT_PATH}")


if __name__ == "__main__":
    parse_saved_html()
