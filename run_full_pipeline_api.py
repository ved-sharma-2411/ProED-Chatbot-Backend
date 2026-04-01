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
from urllib.parse import urlencode

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


BASE_URL = "https://www.ecfr.gov"
TITLE = "34"
DEFAULT_PARTS = ["600", "674", "675", "676", "668", "682", "685", "686", "690"]
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "proed-chatbot")
PARA_MARKERS_RE = re.compile(r"^((?:\([^)]+\)\s*)+)")
SINGLE_MARKER_RE = re.compile(r"\(([^)]+)\)")


def clean_text(text: str) -> str:
    text = " ".join((text or "").split())
    text = re.sub(r"\(\s*([^)]+?)\s*\)", lambda m: f"({m.group(1).strip()})", text)
    text = re.sub(r"\s+([,;:.])", r"\1", text)
    return text.strip()


def build_source_name(url: str) -> str:
    tail = url.split("/")[-1] if "/" in url else url
    return f"ecfr.gov::{tail}"


def get_title_issue_date(client: httpx.Client, title: str) -> str:
    resp = client.get(f"{BASE_URL}/api/versioner/v1/titles.json", timeout=30.0)
    resp.raise_for_status()
    data = resp.json()
    for t in data.get("titles", []):
        if str(t.get("number")) == str(title):
            issue = str(t.get("latest_issue_date") or "").strip()
            if issue:
                return issue
    raise RuntimeError(f"Could not determine latest_issue_date for title {title}")


def get_sections_for_part(client: httpx.Client, title: str, part: str) -> List[Dict[str, str]]:
    resp = client.get(f"{BASE_URL}/api/versioner/v1/versions/title-{title}.json", timeout=60.0)
    resp.raise_for_status()
    rows = resp.json().get("content_versions", [])

    unique: Dict[str, Dict[str, str]] = {}
    for row in rows:
        if str(row.get("type")) != "section":
            continue
        if str(row.get("part")) != str(part):
            continue
        if not bool(row.get("substantive", True)):
            continue
        if bool(row.get("removed", False)):
            continue

        section_id = str(row.get("identifier") or "").strip()
        if not section_id:
            continue

        # Keep latest entry encountered for section identifier.
        unique[section_id] = {
            "identifier": section_id,
            "name": str(row.get("name") or "").strip(),
        }

    return sorted(unique.values(), key=lambda x: x["identifier"])


def fetch_section_xml(client: httpx.Client, issue_date: str, title: str, part: str, section_id: str) -> str:
    query = urlencode({"part": part, "section": section_id})
    url = f"{BASE_URL}/api/versioner/v1/full/{issue_date}/title-{title}.xml?{query}"
    resp = client.get(url, timeout=60.0)
    resp.raise_for_status()
    return resp.text


def build_subsection_tree(paragraphs: List[Dict[str, str]]) -> List[Dict[str, object]]:
    root = {"id": None, "label": None, "text": "", "children": []}
    path_index = {(): root}

    for item in paragraphs:
        node_id = item.get("id") or ""
        parts = re.findall(r"\(([^)]+)\)", node_id)
        path = ()
        for part in parts[1:]:  # skip section token from p-<section>
            parent = path_index[path]
            path = (*path, part)
            if path not in path_index:
                node = {"id": None, "label": part, "text": "", "children": []}
                parent["children"].append(node)
                path_index[path] = node

        target = path_index[path]
        target["id"] = node_id
        target["text"] = item.get("text", "")

    return root["children"]


def parse_section_xml(xml_text: str, section_id: str, part: str, issue_date: str) -> Dict[str, object]:
    soup = BeautifulSoup(xml_text, "xml")
    head = soup.find("HEAD")
    heading = clean_text(head.get_text(" ", strip=True) if head else f"§ {section_id}")

    section_url = f"{BASE_URL}/current/title-34/section-{section_id}"

    paragraphs: List[Dict[str, str]] = []
    para_nodes = soup.find_all("P")
    for idx, p in enumerate(para_nodes, start=1):
        text = clean_text(p.get_text(" ", strip=True))
        if not text:
            continue

        m = PARA_MARKERS_RE.match(text)
        if m:
            parts = SINGLE_MARKER_RE.findall(m.group(1))
            node_id = f"p-{section_id}" + "".join(f"({x})" for x in parts)
        else:
            # Lead paragraph without marker, keep deterministic id.
            node_id = f"p-{section_id}"
            if any(x.get("id") == node_id for x in paragraphs):
                node_id = f"p-{section_id}(p{idx})"

        paragraphs.append({"id": node_id, "text": text})

    return {
        "section": section_id,
        "heading": heading,
        "paragraphs": paragraphs,
        "subsections": build_subsection_tree(paragraphs),
        "source_name": build_source_name(section_url),
        "source_url": section_url,
        "source_type": "cfr",
        "authority_level": "regulation",
        "document_title": heading,
        "section_ref": f"§{section_id}",
        "effective_date": issue_date,
        "ingestion_date": date.today().isoformat(),
        "volume": "",
        "chapter": "VI",
        "title": "34",
        "part": part,
    }


def run_pipeline(
    parts: List[str],
    output_dir: str,
    namespace: str,
    embedding_model: str,
    index_name: str,
    min_tokens: int,
    max_tokens: int,
    overlap_tokens: int,
    skip_ingest: bool,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    configure_quiet_ml_runtime()
    parts_root = os.path.join(output_dir, "parts_api")
    os.makedirs(parts_root, exist_ok=True)

    load_dotenv()
    os.environ["EMBEDDING_MODEL"] = embedding_model
    print(f"Using embedding model: {embedding_model}")
    print(f"Using Pinecone index: {index_name}")

    all_chunk_files: List[str] = []

    with httpx.Client(follow_redirects=True, timeout=60.0) as client:
        issue_date = get_title_issue_date(client, TITLE)
        print(f"Using eCFR issue date: {issue_date}")

        for part in parts:
            part = str(part).strip()
            if not part:
                continue

            part_dir = os.path.join(parts_root, f"part-{part}")
            os.makedirs(part_dir, exist_ok=True)

            print(f"\nFetching part {part} from API...")
            section_rows = get_sections_for_part(client, TITLE, part)
            if not section_rows:
                print(f"[WARN] No sections found for part {part}")
                continue

            print(f"Found {len(section_rows)} sections in part {part}")

            part_sections: List[Dict[str, object]] = []
            part_base_chunks: List[Dict[str, object]] = []
            part_level_chunks: List[Dict[str, object]] = []

            for row in section_rows:
                section_id = row["identifier"]
                try:
                    xml_text = fetch_section_xml(client, issue_date, TITLE, part, section_id)
                    section_data = parse_section_xml(xml_text, section_id, part, issue_date)
                    chunked = build_logical_chunks(
                        section_data,
                        min_tokens=min_tokens,
                        max_tokens=max_tokens,
                        overlap_tokens=overlap_tokens,
                    )
                    part_sections.append(section_data)
                    part_base_chunks.extend(chunked.get("base_chunks", []))
                    part_level_chunks.extend(chunked.get("level_chunks", []))
                    print(f"  processed section {section_id}")
                except Exception as exc:
                    print(f"[WARN] section {section_id} failed: {exc}")

            if not part_sections:
                print(f"[WARN] No valid section content for part {part}")
                continue

            sections_json = os.path.join(part_dir, f"part-{part}_sections.json")
            chunks_json = os.path.join(part_dir, f"part-{part}_chunks.json")
            manifest_json = os.path.join(part_dir, f"part-{part}_manifest.json")

            with open(sections_json, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "part": part,
                        "part_url": f"{BASE_URL}/current/title-34/subtitle-B/chapter-VI/part-{part}",
                        "source_type": "cfr",
                        "authority_level": "regulation",
                        "issue_date": issue_date,
                        "ingestion_date": date.today().isoformat(),
                        "sections": part_sections,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

            with open(chunks_json, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "part": part,
                        "part_url": f"{BASE_URL}/current/title-34/subtitle-B/chapter-VI/part-{part}",
                        "source_name": f"ecfr.gov::part-{part}",
                        "source_url": f"{BASE_URL}/current/title-34/subtitle-B/chapter-VI/part-{part}",
                        "source_type": "cfr",
                        "authority_level": "regulation",
                        "document_title": f"34 CFR Part {part}",
                        "issue_date": issue_date,
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

            with open(manifest_json, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "part": part,
                        "files": {
                            "sections_json": sections_json,
                            "chunks_json": chunks_json,
                        },
                        "counts": {
                            "sections": len(part_sections),
                            "base_chunks": len(part_base_chunks),
                            "level_chunks": len(part_level_chunks),
                        },
                        "issue_date": issue_date,
                        "embedding_model": embedding_model,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

            all_chunk_files.append(chunks_json)
            print(f"Part {part} complete: sections={len(part_sections)} base_chunks={len(part_base_chunks)}")

    if skip_ingest:
        print("\nIngest skipped (--skip-ingest enabled).")
        print(f"Generated part chunk files: {len(all_chunk_files)}")
        return

    print(f"\nIngesting {len(all_chunk_files)} part chunk files into namespace '{namespace}'...")
    for path in all_chunk_files:
        try:
            ingest_chunks(
                input_path=path,
                namespace=namespace,
                chunk_source="base_chunks",
                index_name=index_name,
                embedding_model=embedding_model,
            )
        except Exception as exc:
            print(f"[WARN] Ingest failed for {path}: {exc}")

    print("\nAPI pipeline completed.")


def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run eCFR API -> parse -> chunk -> ingest pipeline")
    p.add_argument("--parts", nargs="+", default=DEFAULT_PARTS)
    p.add_argument("--output-dir", default="data/raw_html/ecfr")
    p.add_argument("--namespace", default="default")
    p.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    p.add_argument("--index-name", default=DEFAULT_INDEX_NAME)
    p.add_argument("--min-tokens", type=int, default=200)
    p.add_argument("--max-tokens", type=int, default=1400)
    p.add_argument("--overlap-tokens", type=int, default=200)
    p.add_argument("--skip-ingest", action="store_true")
    return p


def main() -> None:
    args = build_cli().parse_args()
    run_pipeline(
        parts=args.parts,
        output_dir=args.output_dir,
        namespace=args.namespace,
        embedding_model=args.embedding_model,
        index_name=args.index_name,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap_tokens,
        skip_ingest=args.skip_ingest,
    )


if __name__ == "__main__":
    main()
