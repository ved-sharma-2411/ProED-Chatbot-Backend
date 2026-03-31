import os
import re
import json
import argparse
from statistics import median
from collections import defaultdict

try:
    import tiktoken
except ImportError:
    tiktoken = None

SECTION_ID_RE = re.compile(r"^p-(?P<section>\d+(?:\.\d+)+)(?P<suffix>(?:\([^)]+\))*)$")


def normalize_source_type(source_type: str, source_url: str) -> str:
    st = (source_type or "").strip().lower()
    if st in {"website", "web", "html"}:
        low_url = (source_url or "").lower()
        if "ecfr.gov" in low_url:
            return "cfr"
    return st or "unknown"


def normalize_authority_level(authority_level: str, source_type: str) -> str:
    al = (authority_level or "").strip().lower()
    st = (source_type or "").strip().lower()
    if al == "high":
        return "regulation" if st == "cfr" else "statute"
    if al == "medium":
        return "regulation"
    if al == "low":
        return "sub_regulatory"
    if al in {"statute", "regulation", "sub_regulatory"}:
        return al
    return "unknown"


def build_section_ref(section: str, path: list) -> str:
    section = str(section or "").strip()
    if not section or section == "unknown":
        return ""

    suffix_parts = []
    for p in path or []:
        token = str(p).strip()
        if not token or token.startswith("p"):
            continue
        suffix_parts.append(f"({token})")

    return f"§{section}{''.join(suffix_parts)}"


def get_token_codec():
    if tiktoken is not None:
        enc = tiktoken.get_encoding("cl100k_base")
        return {
            "name": "tiktoken/cl100k_base",
            "encode": lambda text: enc.encode(text or ""),
            "decode": lambda tokens: enc.decode(tokens or []),
        }

    # Fallback tokenizer if tiktoken is unavailable
    return {
        "name": "regex-fallback",
        "encode": lambda text: re.findall(r"\w+|[^\w\s]", text or ""),
        "decode": lambda tokens: " ".join(tokens or []),
    }


def count_tokens(text: str, encode_fn) -> int:
    return len(encode_fn(text or ""))


def attach_token_counts(chunks, encode_fn):
    for c in chunks:
        c["char_count"] = len(c.get("text", ""))
        c["token_count"] = count_tokens(c.get("text", ""), encode_fn)
    return chunks


def apply_chunk_overlap(chunks, overlap_tokens, encode_fn, decode_fn, tokenizer_name):
    if not chunks:
        return chunks

    originals = [c.get("text", "") for c in chunks]
    total = len(chunks)

    for i, chunk in enumerate(chunks):
        prev_id = chunks[i - 1]["chunk_id"] if i > 0 else None
        next_id = chunks[i + 1]["chunk_id"] if i < total - 1 else None

        overlap_text = ""
        overlap_used = 0

        if i > 0 and overlap_tokens > 0:
            prev_tokens = encode_fn(originals[i - 1])
            if prev_tokens:
                tail = prev_tokens[-overlap_tokens:]
                overlap_used = len(tail)
                overlap_text = (decode_fn(tail) or "").strip()

        body_text = originals[i].strip()
        if overlap_text:
            combined = f"{overlap_text}\n\n{body_text}".strip()
        else:
            combined = body_text

        chunk["overlap_text"] = overlap_text
        chunk["overlap_tokens"] = overlap_used
        chunk["text_no_overlap"] = body_text
        chunk["text"] = combined
        chunk["embedding_text"] = combined
        chunk["char_count"] = len(combined)
        chunk["token_count"] = count_tokens(combined, encode_fn)
        chunk["metadata"] = {
            "chunk_index": i + 1,
            "total_chunks": total,
            "prev_chunk_id": prev_id,
            "next_chunk_id": next_id,
            "tokenizer": tokenizer_name,
            "overlap_from_previous_tokens": overlap_used,
            "has_overlap": bool(overlap_text),
        }

    return chunks


def parse_id_parts(node_id: str):
    if not node_id:
        return None, []

    match = SECTION_ID_RE.match(node_id)
    if not match:
        return None, []

    section = match.group("section")
    suffix = match.group("suffix")
    parts = re.findall(r"\(([^)]+)\)", suffix)
    return section, parts


def normalize_text(text: str) -> str:
    return " ".join((text or "").split()).strip()


def build_tree(paragraphs):
    root = {
        "path": (),
        "label": None,
        "id": None,
        "text": "",
        "children": {},
        "parent": None,
    }
    index = {(): root}
    section = ""

    for item in paragraphs:
        node_id = item.get("id")
        text = normalize_text(item.get("text", ""))
        node_section, parts = parse_id_parts(node_id)
        if node_section and not section:
            section = node_section

        path = ()
        for part in parts:
            parent = index[path]
            path = (*path, part)
            if path not in index:
                index[path] = {
                    "path": path,
                    "label": part,
                    "id": None,
                    "text": "",
                    "children": {},
                    "parent": parent,
                }
                parent["children"][part] = index[path]

        node = index[path]
        if node_id:
            node["id"] = node_id
        if text:
            node["text"] = text

    return root, index, section


def aggregate_node_text(node):
    base = normalize_text(node.get("text", ""))
    chunks = [base] if base else []

    for key in sorted(node["children"].keys(), key=str):
        child = node["children"][key]
        child_text = aggregate_node_text(child)
        if not child_text:
            continue

        low_current = " ".join(chunks).lower()
        low_child = child_text.lower()

        if low_child in low_current:
            continue

        chunks.append(child_text)

    return "\n".join(chunks).strip()


def traverse_nodes(root):
    stack = [root]
    while stack:
        node = stack.pop()
        children = [node["children"][k] for k in sorted(node["children"].keys(), key=str)]
        for child in reversed(children):
            stack.append(child)
        if node["path"]:
            yield node


def make_level_chunks(section, heading, root):
    chunks = []
    for node in traverse_nodes(root):
        combined = aggregate_node_text(node)
        if not combined:
            continue

        depth = len(node["path"])
        label_path = " > ".join(node["path"])
        parent_path = node["path"][:-1]
        chunk_id = f"{section}:{'/'.join(node['path'])}"

        chunks.append(
            {
                "chunk_id": chunk_id,
                "section": section,
                "section_ref": build_section_ref(section, list(node["path"])),
                "heading": heading,
                "depth": depth,
                "path": list(node["path"]),
                "path_label": label_path,
                "parent_path": list(parent_path),
                "node_id": node.get("id"),
                "self_text": node.get("text", ""),
                "text": combined,
                "char_count": len(combined),
                "source_name": "",
                "source_url": "",
                "source_type": "unknown",
                "authority_level": "unknown",
                "document_title": "",
                "effective_date": "",
                "ingestion_date": "",
                "volume": "",
                "chapter": "",
                "title": "",
                "part": "",
            }
        )

    return chunks


def detect_best_depth(level_chunks):
    by_depth = defaultdict(list)
    for c in level_chunks:
        by_depth[c["depth"]].append(c)

    if not by_depth:
        return 1, {}

    stats = {}
    target = 800

    best_depth = None
    best_score = float("inf")

    for depth in sorted(by_depth.keys()):
        sizes = [c["token_count"] for c in by_depth[depth] if c["token_count"] > 0]
        if not sizes:
            continue

        med = median(sizes)
        avg = sum(sizes) / len(sizes)

        score = abs(med - target)
        if med < 200:
            score += 500
        if med > 1500:
            score += 700
        if len(sizes) == 1:
            score += 100

        stats[depth] = {
            "count": len(sizes),
            "median_tokens": int(med),
            "avg_tokens": int(avg),
            "score": int(score),
        }

        if score < best_score:
            best_score = score
            best_depth = depth

    if best_depth is None:
        best_depth = min(by_depth.keys())

    return best_depth, stats


def merge_small_siblings(chunks, min_tokens, max_tokens):
    grouped = defaultdict(list)
    for c in chunks:
        grouped[tuple(c["parent_path"])].append(c)

    merged = []
    for parent, siblings in grouped.items():
        siblings = sorted(siblings, key=lambda x: x["path"])
        if len(siblings) == 1:
            merged.append(siblings[0])
            continue

        avg_tokens = sum(s["token_count"] for s in siblings) / len(siblings)

        if avg_tokens >= min_tokens:
            merged.extend(siblings)
            continue

        buffer = []
        buffer_tokens = 0
        for s in siblings:
            if not buffer:
                buffer = [s]
                buffer_tokens = s["token_count"]
                continue

            if buffer_tokens < min_tokens and (buffer_tokens + s["token_count"]) <= max_tokens:
                buffer.append(s)
                buffer_tokens += s["token_count"]
            else:
                merged.append(_finalize_merged_group(buffer))
                buffer = [s]
                buffer_tokens = s["token_count"]

        if buffer:
            merged.append(_finalize_merged_group(buffer))

    return sorted(merged, key=lambda x: x["path"])


def _finalize_merged_group(group):
    if len(group) == 1:
        return group[0]

    first = group[0]
    merged_text = "\n\n".join(c["text"] for c in group if c["text"].strip())

    return {
        "chunk_id": f"{first['chunk_id']}+{len(group)-1}",
        "section": first["section"],
        "section_ref": first.get("section_ref", ""),
        "heading": first["heading"],
        "depth": first["depth"],
        "path": first["path"],
        "path_label": f"{first['path_label']} (merged {len(group)} siblings)",
        "parent_path": first["parent_path"],
        "node_id": first.get("node_id"),
        "self_text": "",
        "text": merged_text,
        "char_count": len(merged_text),
        "member_chunk_ids": [c["chunk_id"] for c in group],
        "source_name": first.get("source_name", ""),
        "source_url": first.get("source_url", ""),
        "source_type": first.get("source_type", "unknown"),
        "authority_level": first.get("authority_level", "unknown"),
        "document_title": first.get("document_title", ""),
        "effective_date": first.get("effective_date", ""),
        "ingestion_date": first.get("ingestion_date", ""),
        "volume": first.get("volume", ""),
        "chapter": first.get("chapter", ""),
        "title": first.get("title", ""),
        "part": first.get("part", ""),
    }


def split_text_by_tokens(text, max_tokens, encode_fn):
    text = (text or "").strip()
    if not text:
        return []

    if count_tokens(text, encode_fn) <= max_tokens:
        return [text]

    segments = [
        s.strip()
        for s in re.split(r"(?<=[\.!\?;])\s+|\n+", text)
        if s and s.strip()
    ]
    if not segments:
        segments = [text]

    chunks = []
    buf = []

    for seg in segments:
        seg_tokens = count_tokens(seg, encode_fn)

        if seg_tokens > max_tokens:
            if buf:
                chunks.append(" ".join(buf).strip())
                buf = []
            chunks.extend(_hard_split_segment(seg, max_tokens, encode_fn))
            continue

        candidate = " ".join(buf + [seg]).strip()
        if candidate and count_tokens(candidate, encode_fn) <= max_tokens:
            buf.append(seg)
        else:
            if buf:
                chunks.append(" ".join(buf).strip())
            buf = [seg]

    if buf:
        chunks.append(" ".join(buf).strip())

    return [c for c in chunks if c]


def _hard_split_segment(segment, max_tokens, encode_fn):
    words = segment.split()
    out = []
    buf = []

    for word in words:
        candidate = " ".join(buf + [word]).strip()
        if not buf or count_tokens(candidate, encode_fn) <= max_tokens:
            buf.append(word)
        else:
            out.append(" ".join(buf).strip())
            buf = [word]

    if buf:
        out.append(" ".join(buf).strip())

    return out


def split_oversized_chunks(chunks, max_tokens, encode_fn):
    out = []

    for chunk in chunks:
        tokens = chunk["token_count"]
        if tokens <= max_tokens:
            out.append(chunk)
            continue

        parts = split_text_by_tokens(chunk.get("text", ""), max_tokens, encode_fn)
        if len(parts) <= 1:
            out.append(chunk)
            continue

        for i, part_text in enumerate(parts, start=1):
            new_chunk = dict(chunk)
            new_chunk["chunk_id"] = f"{chunk['chunk_id']}::part{i}"
            new_chunk["path_label"] = f"{chunk['path_label']} (part {i}/{len(parts)})"
            new_chunk["text"] = part_text
            new_chunk["self_text"] = ""
            out.append(new_chunk)

    return attach_token_counts(out, encode_fn)


def merge_under_min_chunks(chunks, min_tokens, max_tokens, encode_fn):
    grouped = defaultdict(list)
    for c in chunks:
        grouped[(tuple(c["parent_path"]), c["depth"])].append(c)

    merged = []
    for _key, siblings in grouped.items():
        siblings = sorted(siblings, key=lambda x: x["path"])
        buffer = []
        buffer_tokens = 0

        for s in siblings:
            s_tokens = s["token_count"]
            if not buffer:
                buffer = [s]
                buffer_tokens = s_tokens
                continue

            if buffer_tokens < min_tokens and (buffer_tokens + s_tokens) <= max_tokens:
                buffer.append(s)
                buffer_tokens += s_tokens
            else:
                merged.append(_finalize_merged_group(buffer))
                buffer = [s]
                buffer_tokens = s_tokens

        if buffer:
            merged.append(_finalize_merged_group(buffer))

    merged = attach_token_counts(merged, encode_fn)

    # Last pass: if final small chunk can be merged with previous sibling, merge it.
    finalized = []
    for c in sorted(merged, key=lambda x: (x["depth"], x["path"])):
        if (
            finalized
            and c["token_count"] < min_tokens
            and finalized[-1]["parent_path"] == c["parent_path"]
            and finalized[-1]["depth"] == c["depth"]
            and finalized[-1]["token_count"] + c["token_count"] <= max_tokens
        ):
            finalized[-1] = _finalize_merged_group([finalized[-1], c])
            finalized[-1]["token_count"] = count_tokens(finalized[-1]["text"], encode_fn)
            finalized[-1]["char_count"] = len(finalized[-1]["text"])
        else:
            finalized.append(c)

    return finalized


def enforce_token_limits(chunks, min_tokens, max_tokens, encode_fn):
    chunks = attach_token_counts(chunks, encode_fn)
    chunks = split_oversized_chunks(chunks, max_tokens=max_tokens, encode_fn=encode_fn)
    chunks = merge_under_min_chunks(
        chunks,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        encode_fn=encode_fn,
    )
    chunks = split_oversized_chunks(chunks, max_tokens=max_tokens, encode_fn=encode_fn)
    return sorted(chunks, key=lambda x: (x["depth"], x["path"]))


def fallback_chunks_from_paragraphs(section, heading, paragraphs):
    chunks = []
    for idx, p in enumerate(paragraphs, start=1):
        text = normalize_text(p.get("text", ""))
        if not text:
            continue

        node_id = p.get("id")
        _, parts = parse_id_parts(node_id)
        path = parts or [f"p{idx}"]
        chunks.append(
            {
                "chunk_id": f"{section}:{'/'.join(path)}",
                "section": section,
                "section_ref": build_section_ref(section, path),
                "heading": heading,
                "depth": len(path),
                "path": path,
                "path_label": " > ".join(path),
                "parent_path": path[:-1],
                "node_id": node_id,
                "self_text": text,
                "text": text,
                "char_count": len(text),
                "source_name": "",
                "source_url": "",
                "source_type": "unknown",
                "authority_level": "unknown",
                "document_title": "",
                "effective_date": "",
                "ingestion_date": "",
                "volume": "",
                "chapter": "",
                "title": "",
                "part": "",
            }
        )
    return chunks


def attach_source_fields(
    chunks,
    source_name,
    source_url,
    source_type,
    authority_level,
    document_title="",
    effective_date="",
    ingestion_date="",
    volume="",
    chapter="",
    title="",
    part="",
):
    for c in chunks:
        c["source_name"] = source_name or c.get("source_name", "")
        c["source_url"] = source_url or c.get("source_url", "")
        c["source_type"] = source_type or c.get("source_type", "unknown")
        c["authority_level"] = authority_level or c.get("authority_level", "unknown")
        c["document_title"] = document_title or c.get("document_title", "")
        c["effective_date"] = effective_date or c.get("effective_date", "")
        c["ingestion_date"] = ingestion_date or c.get("ingestion_date", "")
        c["volume"] = volume or c.get("volume", "")
        c["chapter"] = chapter or c.get("chapter", "")
        c["title"] = title or c.get("title", "")
        c["part"] = part or c.get("part", "")
    return chunks


def build_logical_chunks(data, min_tokens=200, max_tokens=1400, overlap_tokens=200):
    section = str(data.get("section", "")).strip()
    heading = data.get("heading", "")
    paragraphs = data.get("paragraphs", [])
    source_name = str(data.get("source_name") or "eCFR 34 CFR 668.32")
    source_url = str(data.get("source_url") or "")
    source_type = normalize_source_type(str(data.get("source_type") or "cfr"), source_url)
    authority_level = normalize_authority_level(str(data.get("authority_level") or "regulation"), source_type)
    document_title = str(data.get("document_title") or heading or f"34 CFR §{section}".strip())
    effective_date = str(data.get("effective_date") or "")
    ingestion_date = str(data.get("ingestion_date") or "")
    volume = str(data.get("volume") or "")
    chapter = str(data.get("chapter") or "")
    title = str(data.get("title") or "")
    part = str(data.get("part") or "")
    token_codec = get_token_codec()
    encode_fn = token_codec["encode"]
    decode_fn = token_codec["decode"]
    tokenizer_name = token_codec["name"]

    blocked_markers = (
        "aggressive automated scraping",
        "complete the captcha",
        "request access",
        "programmatic access to these sites is limited",
    )
    joined_text = "\n".join(str(p.get("text", "")) for p in paragraphs).lower()
    if any(m in joined_text for m in blocked_markers):
        raise RuntimeError(
            "Detected anti-bot/CAPTCHA content in parsed input. "
            "Stop ingestion and refresh source HTML first."
        )

    root, _index, inferred_section = build_tree(paragraphs)
    if not section:
        section = inferred_section or "unknown"

    level_chunks = make_level_chunks(section, heading, root)

    if not level_chunks:
        level_chunks = fallback_chunks_from_paragraphs(section, heading, paragraphs)

    level_chunks = attach_source_fields(
        level_chunks,
        source_name=source_name,
        source_url=source_url,
        source_type=source_type,
        authority_level=authority_level,
        document_title=document_title,
        effective_date=effective_date,
        ingestion_date=ingestion_date,
        volume=volume,
        chapter=chapter,
        title=title,
        part=part,
    )

    level_chunks = attach_token_counts(level_chunks, encode_fn)

    best_depth, depth_stats = detect_best_depth(level_chunks)

    base = [c for c in level_chunks if c["depth"] == best_depth]
    if not base:
        deepest = max((c["depth"] for c in level_chunks), default=1)
        base = [c for c in level_chunks if c["depth"] == deepest]
        best_depth = deepest

    base = merge_small_siblings(base, min_tokens=min_tokens, max_tokens=max_tokens)
    base = attach_token_counts(base, encode_fn)
    base = enforce_token_limits(
        base,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        encode_fn=encode_fn,
    )
    base = apply_chunk_overlap(
        base,
        overlap_tokens=overlap_tokens,
        encode_fn=encode_fn,
        decode_fn=decode_fn,
        tokenizer_name=tokenizer_name,
    )
    for c in base:
        c["self_text"] = ""
        c["embedding_text"] = c.get("embedding_text") or c.get("text", "")
    base = attach_source_fields(
        base,
        source_name=source_name,
        source_url=source_url,
        source_type=source_type,
        authority_level=authority_level,
        document_title=document_title,
        effective_date=effective_date,
        ingestion_date=ingestion_date,
        volume=volume,
        chapter=chapter,
        title=title,
        part=part,
    )

    by_depth = defaultdict(list)
    for c in level_chunks:
        by_depth[c["depth"]].append(c)

    level_summary = {
        str(depth): {
            "count": len(items),
            "total_chars": sum(x["char_count"] for x in items),
            "total_tokens": sum(x["token_count"] for x in items),
            "avg_chars": int(sum(x["char_count"] for x in items) / max(len(items), 1)),
            "avg_tokens": int(sum(x["token_count"] for x in items) / max(len(items), 1)),
        }
        for depth, items in sorted(by_depth.items())
    }

    base_tokens = [c["token_count"] for c in base if c["token_count"] > 0]

    return {
        "section": section,
        "heading": heading,
        "source_name": source_name,
        "source_url": source_url,
        "source_type": source_type,
        "authority_level": authority_level,
        "document_title": document_title,
        "section_ref": f"§{section}" if section and section != "unknown" else "",
        "effective_date": effective_date,
        "ingestion_date": ingestion_date,
        "volume": volume,
        "chapter": chapter,
        "title": title,
        "part": part,
        "rules": {
            "id_hierarchy": "(a) -> parent, (a)(1) -> child, (a)(1)(i) -> deeper",
            "best_depth_selected": best_depth,
            "depth_stats": depth_stats,
            "token_size_control": {
                "enabled": True,
                "tokenizer": tokenizer_name,
                "min_tokens": min_tokens,
                "max_tokens": max_tokens,
            },
            "chunk_overlap": {
                "enabled": True,
                "overlap_tokens": overlap_tokens,
                "strategy": "prepend previous chunk tail to current chunk",
            },
            "small_sibling_merge": {
                "enabled": True,
                "min_tokens": min_tokens,
                "max_tokens": max_tokens,
            },
        },
        "level_summary": level_summary,
        "base_summary": {
            "count": len(base),
            "min_tokens": min(base_tokens) if base_tokens else 0,
            "max_tokens": max(base_tokens) if base_tokens else 0,
            "avg_tokens": int(sum(base_tokens) / len(base_tokens)) if base_tokens else 0,
        },
        "level_chunks": sorted(level_chunks, key=lambda x: (x["depth"], x["path"])),
        "base_chunks": base,
    }


def output_paths(input_json_path: str):
    base, _ = os.path.splitext(input_json_path)
    return f"{base}_chunks.json", f"{base}_base_chunks.txt"


def process_file(
    input_json_path: str,
    min_tokens: int,
    max_tokens: int,
    overlap_tokens: int,
):
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunked = build_logical_chunks(
        data,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )
    out_json, out_txt = output_paths(input_json_path)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(chunked, f, indent=2, ensure_ascii=False)

    with open(out_txt, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunked["base_chunks"], start=1):
            f.write(f"=== BASE CHUNK {i} ===\n")
            f.write(f"chunk_id: {chunk['chunk_id']}\n")
            f.write(f"path: {chunk['path_label']}\n")
            f.write(f"chars: {chunk['char_count']}\n\n")
            f.write(f"tokens: {chunk.get('token_count', 0)}\n\n")
            f.write(f"overlap_tokens: {chunk.get('overlap_tokens', 0)}\n")
            if chunk.get("metadata"):
                f.write(f"prev_chunk_id: {chunk['metadata'].get('prev_chunk_id')}\n")
                f.write(f"next_chunk_id: {chunk['metadata'].get('next_chunk_id')}\n\n")
            f.write(chunk["text"].strip() + "\n\n")

    print(f"Chunked JSON: {out_json}")
    print(f"Base chunks TXT: {out_txt}")


def build_cli():
    parser = argparse.ArgumentParser(
        description="Build logical chunks from parsed ECFR HTML JSON"
    )
    parser.add_argument(
        "--input",
        default="data/raw_html/ecfr/section_668_32.json",
        help="Path to parsed HTML JSON",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=200,
        help="Minimum tokens per base chunk",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1400,
        help="Maximum tokens per base chunk (recommended 1200-1500)",
    )
    parser.add_argument(
        "--overlap-tokens",
        type=int,
        default=200,
        help="Token overlap between consecutive base chunks",
    )
    return parser


if __name__ == "__main__":
    args = build_cli().parse_args()
    process_file(
        args.input,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap_tokens,
    )
