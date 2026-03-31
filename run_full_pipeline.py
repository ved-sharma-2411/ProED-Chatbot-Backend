from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import re
import warnings
from datetime import date
from typing import Dict, List, Tuple
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from rag_pinecone import ingest_chunks, load_dotenv


def configure_quiet_ml_runtime() -> None:
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    warnings.filterwarnings(
        "ignore",
        message=r".*You are sending unauthenticated requests to the HF Hub.*",
    )

    for noisy in [
        "sentence_transformers",
        "transformers",
        "huggingface_hub",
        "huggingface_hub.utils._http",
        "model2vec",
    ]:
        logging.getLogger(noisy).setLevel(logging.ERROR)


def _load_build_logical_chunks():
    script_path = os.path.join(os.path.dirname(__file__), "html-chunk.py")
    spec = importlib.util.spec_from_file_location("html_chunk_runtime", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "build_logical_chunks"):
        raise ImportError("build_logical_chunks not found in html-chunk.py")
    return module.build_logical_chunks


build_logical_chunks = _load_build_logical_chunks()


ECFR_BASE = "https://www.ecfr.gov"
DEFAULT_PARTS = ["600", "674", "675", "676", "668", "682", "685", "686", "690"]
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SECTION_ID_RE = re.compile(r"^p-(?P<section>\d+(?:\.\d+)+)(?P<suffix>(?:\([^)]+\))*)$")
SECTION_URL_RE = re.compile(r"/section-(\d+(?:\.\d+)+)")
BLOCK_MARKERS = (
    "aggressive automated scraping",
    "complete the captcha",
    "request access",
    "programmatic access to these sites is limited",
)


def slugify_url(url: str) -> str:
    parsed = urlparse(url)
    tail = parsed.path.strip("/").split("/")[-1] or "page"
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", tail)


def clean_text(text: str) -> str:
    text = " ".join((text or "").split())
    text = re.sub(r"\(\s*([^)]+?)\s*\)", lambda m: f"({m.group(1).strip()})", text)
    text = re.sub(r"\s+([,;:.])", r"\1", text)
    return text.strip()


def build_source_name(url: str) -> str:
    parsed = urlparse(url)
    host = (parsed.netloc or "source").replace("www.", "")
    tail = parsed.path.strip("/").split("/")[-1] if parsed.path else "page"
    return f"{host}::{tail}"


def parse_id_parts(node_id: str) -> Tuple[str | None, List[str]]:
    if not node_id:
        return None, []
    m = SECTION_ID_RE.match(node_id)
    if not m:
        return None, []
    section = m.group("section")
    parts = re.findall(r"\(([^)]+)\)", m.group("suffix"))
    return section, parts


def build_subsection_tree(paragraphs: List[Dict[str, str]]) -> List[Dict[str, object]]:
    root = {"id": None, "label": None, "text": "", "children": []}
    path_index = {(): root}

    for item in paragraphs:
        node_id = item.get("id")
        text = item.get("text", "")
        _, parts = parse_id_parts(node_id or "")

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


def extract_section_links(part_html: str, part_number: str) -> List[str]:
    soup = BeautifulSoup(part_html, "lxml")
    urls = []
    seen = set()
    section_prefix_re = re.compile(rf"/section-{re.escape(part_number)}\.\d+(?:$|[?#])")
    anchor_re = re.compile(rf"^#{re.escape(part_number)}\.\d+(?:$|\b)")

    for a in soup.select("a[href]"):
        href = (a.get("href") or "").strip()
        if not href:
            continue

        full = ""

        # Pattern 1: direct section links (e.g. /current/title-34/section-600.4)
        if "/section-" in href:
            candidate = urljoin(ECFR_BASE, href)
            if section_prefix_re.search(candidate):
                full = candidate.split("#", 1)[0]

        # Pattern 2: in-page anchors (e.g. #600.4) on part pages
        elif href.startswith("#") and anchor_re.match(href):
            section_id = href[1:]
            full = f"{ECFR_BASE}/current/title-34/section-{section_id}"

        if not full or full in seen:
            continue

        seen.add(full)
        urls.append(full)

    return urls


def parse_section_html(html: str, section_url: str) -> Dict[str, object]:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav"]):
        tag.decompose()

    heading = ""
    for h in soup.find_all(["h1", "h2", "h3", "h4", "h5"]):
        t = clean_text(h.get_text(" ", strip=True))
        if "§" in t:
            heading = t
            break

    url_section_match = SECTION_URL_RE.search(section_url)
    target_section = url_section_match.group(1) if url_section_match else ""

    paragraphs: List[Dict[str, str]] = []
    for node in soup.select('[id^="p-"]'):
        node_id = node.get("id")
        section, _ = parse_id_parts(node_id or "")
        if target_section and section != target_section:
            continue

        text = clean_text(node.get_text(" ", strip=True))
        if text:
            paragraphs.append({"id": node_id, "text": text})

    if not paragraphs:
        article = soup.find("article") or soup
        for p in article.find_all("p"):
            text = clean_text(p.get_text(" ", strip=True))
            if text and "ECFR CONTENT" not in text:
                paragraphs.append({"id": None, "text": text})

    doc_title = heading or (f"34 CFR §{target_section}" if target_section else "34 CFR")
    part_match = re.search(r"/part-(\d+)", section_url)
    part = part_match.group(1) if part_match else ""

    return {
        "section": target_section,
        "heading": heading,
        "paragraphs": paragraphs,
        "subsections": build_subsection_tree(paragraphs),
        "source_name": build_source_name(section_url),
        "source_url": section_url,
        "source_type": "cfr",
        "authority_level": "regulation",
        "document_title": doc_title,
        "section_ref": f"§{target_section}" if target_section else "",
        "effective_date": "",
        "ingestion_date": date.today().isoformat(),
        "volume": "",
        "chapter": "VI",
        "title": "34",
        "part": part,
    }


def fetch_html(client: httpx.Client, url: str) -> str:
    r = client.get(url)
    r.raise_for_status()
    html = r.text
    low = html.lower()
    if any(marker in low for marker in BLOCK_MARKERS):
        raise RuntimeError(f"Blocked/CAPTCHA content detected for {url}")
    return html


def process_section(
    client: httpx.Client,
    section_url: str,
    min_tokens: int,
    max_tokens: int,
    overlap_tokens: int,
) -> Dict[str, object]:
    html = fetch_html(client, section_url)
    data = parse_section_html(html, section_url)

    chunked = build_logical_chunks(
        data,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )

    print(f"Processed: {section_url}")
    return {
        "section_url": section_url,
        "section": data,
        "chunked": chunked,
    }


def run_pipeline(
    parts: List[str],
    output_dir: str,
    namespace: str,
    embedding_model: str,
    min_tokens: int,
    max_tokens: int,
    overlap_tokens: int,
    skip_ingest: bool,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    configure_quiet_ml_runtime()
    parts_root = os.path.join(output_dir, "parts")
    os.makedirs(parts_root, exist_ok=True)
    load_dotenv()
    os.environ["EMBEDDING_MODEL"] = embedding_model
    print(f"Using embedding model: {os.environ['EMBEDDING_MODEL']}")

    all_chunk_files: List[str] = []
    with httpx.Client(follow_redirects=True, timeout=60.0) as client:
        for part in parts:
            part = str(part).strip()
            if not part:
                continue

            part_url = f"{ECFR_BASE}/current/title-34/subtitle-B/chapter-VI/part-{part}"
            print(f"\nScanning part: {part_url}")

            try:
                part_html = fetch_html(client, part_url)
                section_urls = extract_section_links(part_html, part)
            except Exception as exc:
                print(f"[WARN] Failed part {part}: {exc}")
                continue

            if not section_urls:
                print(f"[WARN] No section URLs found for part {part}")
                continue

            part_dir = os.path.join(parts_root, f"part-{part}")
            os.makedirs(part_dir, exist_ok=True)

            part_html_path = os.path.join(part_dir, f"part-{part}.html")
            with open(part_html_path, "w", encoding="utf-8") as f:
                f.write(part_html)

            print(f"Found {len(section_urls)} sections in part {part}")
            part_sections: List[Dict[str, object]] = []
            part_base_chunks: List[Dict[str, object]] = []
            part_level_chunks: List[Dict[str, object]] = []

            for section_url in section_urls:
                try:
                    result = process_section(
                        client=client,
                        section_url=section_url,
                        min_tokens=min_tokens,
                        max_tokens=max_tokens,
                        overlap_tokens=overlap_tokens,
                    )
                    section_data = result["section"]
                    chunked_data = result["chunked"]
                    part_sections.append(section_data)
                    part_base_chunks.extend(chunked_data.get("base_chunks", []))
                    part_level_chunks.extend(chunked_data.get("level_chunks", []))
                except Exception as exc:
                    print(f"[WARN] Failed section {section_url}: {exc}")

            if not part_sections:
                print(f"[WARN] No valid sections parsed for part {part}")
                continue

            part_sections_path = os.path.join(part_dir, f"part-{part}_sections.json")
            with open(part_sections_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "part": part,
                        "part_url": part_url,
                        "source_type": "cfr",
                        "authority_level": "regulation",
                        "ingestion_date": date.today().isoformat(),
                        "sections": part_sections,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

            part_chunks_path = os.path.join(part_dir, f"part-{part}_chunks.json")
            with open(part_chunks_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "part": part,
                        "part_url": part_url,
                        "source_name": build_source_name(part_url),
                        "source_url": part_url,
                        "source_type": "cfr",
                        "authority_level": "regulation",
                        "document_title": f"34 CFR Part {part}",
                        "ingestion_date": date.today().isoformat(),
                        "rules": {
                            "token_size_control": {
                                "min_tokens": min_tokens,
                                "max_tokens": max_tokens,
                            },
                            "chunk_overlap": {
                                "overlap_tokens": overlap_tokens,
                            },
                        },
                        "base_chunks": part_base_chunks,
                        "level_chunks": part_level_chunks,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

            manifest_path = os.path.join(part_dir, f"part-{part}_manifest.json")
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "part": part,
                        "part_url": part_url,
                        "files": {
                            "part_html": part_html_path,
                            "sections_json": part_sections_path,
                            "chunks_json": part_chunks_path,
                        },
                        "counts": {
                            "sections": len(part_sections),
                            "base_chunks": len(part_base_chunks),
                            "level_chunks": len(part_level_chunks),
                        },
                        "embedding_model": embedding_model,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

            print(
                f"Part {part} artifacts -> sections: {len(part_sections)}, "
                f"base_chunks: {len(part_base_chunks)}"
            )
            all_chunk_files.append(part_chunks_path)

    if skip_ingest:
        print("\nIngest skipped (--skip-ingest enabled).")
        print(f"Generated part chunk files: {len(all_chunk_files)}")
        return

    print(f"\nIngesting {len(all_chunk_files)} chunk files into namespace '{namespace}'...")
    for path in all_chunk_files:
        try:
            ingest_chunks(input_path=path, namespace=namespace, chunk_source="base_chunks")
        except Exception as exc:
            print(f"[WARN] Ingest failed for {path}: {exc}")

    print("\nPipeline completed.")
    print(f"Total chunk files generated: {len(all_chunk_files)}")


def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run full eCFR HTML -> parse -> chunk -> ingest pipeline")
    p.add_argument(
        "--parts",
        nargs="+",
        default=DEFAULT_PARTS,
        help="CFR part numbers to scrape (example: 668 674 675)",
    )
    p.add_argument(
        "--output-dir",
        default="data/raw_html/ecfr",
        help="Directory to store raw/parsed/chunk outputs",
    )
    p.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="Embedding model for ingestion (default: sentence-transformers/all-MiniLM-L6-v2)",
    )
    p.add_argument("--namespace", default="default", help="Pinecone namespace")
    p.add_argument("--min-tokens", type=int, default=200)
    p.add_argument("--max-tokens", type=int, default=1400)
    p.add_argument("--overlap-tokens", type=int, default=200)
    p.add_argument("--skip-ingest", action="store_true", help="Only scrape/parse/chunk; do not ingest")
    return p


def main() -> None:
    args = build_cli().parse_args()
    run_pipeline(
        parts=args.parts,
        output_dir=args.output_dir,
        namespace=args.namespace,
        embedding_model=args.embedding_model,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap_tokens,
        skip_ingest=args.skip_ingest,
    )


if __name__ == "__main__":
    main()
