import os
import json
import time
import argparse
from typing import List, Dict, Any

try:
    from openai import OpenAI  # type: ignore[reportMissingImports]
except ImportError:
    OpenAI = None

MODEL = "text-embedding-3-large"


def load_chunks(input_path: str, chunk_source: str = "base_chunks") -> List[Dict[str, Any]]:
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunks = data.get(chunk_source) or []
    if not chunks and chunk_source != "level_chunks":
        chunks = data.get("level_chunks") or []

    if not chunks:
        raise ValueError(f"No chunks found in '{input_path}'. Tried source '{chunk_source}'.")

    return chunks


def text_for_embedding(chunk: Dict[str, Any]) -> str:
    # Prefer non-overlap body if available to avoid repeated context across chunks.
    text = (chunk.get("text_no_overlap") or chunk.get("text") or "").strip()

    heading = (chunk.get("heading") or "").strip()
    path_label = (chunk.get("path_label") or "").strip()

    parts = []
    if heading:
        parts.append(f"Heading: {heading}")
    if path_label:
        parts.append(f"Path: {path_label}")
    if text:
        parts.append(f"Text: {text}")

    return "\n".join(parts).strip()


def batched(seq: List[Any], size: int):
    for i in range(0, len(seq), size):
        yield i, seq[i : i + size]


def create_embeddings(
    chunks: List[Dict[str, Any]],
    model: str,
    batch_size: int,
    max_retries: int,
    retry_sleep: float,
) -> List[Dict[str, Any]]:
    if OpenAI is None:
        raise ImportError("openai package is not installed. Run: pip install openai")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    client = OpenAI(api_key=api_key)

    results = []
    prepared = []
    for idx, chunk in enumerate(chunks):
        prepared.append(
            {
                "index": idx,
                "chunk": chunk,
                "text": text_for_embedding(chunk),
            }
        )

    for start, batch in batched(prepared, batch_size):
        inputs = [x["text"] for x in batch]

        attempt = 0
        while True:
            try:
                response = client.embeddings.create(model=model, input=inputs)
                break
            except Exception:
                attempt += 1
                if attempt > max_retries:
                    raise
                time.sleep(retry_sleep * attempt)

        if len(response.data) != len(batch):
            raise RuntimeError(
                f"Embedding response size mismatch at batch starting {start}: "
                f"expected {len(batch)}, got {len(response.data)}"
            )

        for item, emb in zip(batch, response.data):
            chunk = item["chunk"]
            vector = emb.embedding

            results.append(
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "section": chunk.get("section"),
                    "depth": chunk.get("depth"),
                    "path": chunk.get("path"),
                    "path_label": chunk.get("path_label"),
                    "token_count": chunk.get("token_count"),
                    "model": model,
                    "vector_dim": len(vector),
                    "embedding": vector,
                    "text": item["text"],
                    "metadata": chunk.get("metadata", {}),
                }
            )

        print(f"Embedded {start + len(batch)}/{len(prepared)} chunks")

    return results


def default_output_path(input_path: str) -> str:
    base, _ = os.path.splitext(input_path)
    return f"{base}_embeddings.jsonl"


def write_jsonl(records: List[Dict[str, Any]], output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary(records: List[Dict[str, Any]], output_path: str):
    summary_path = output_path.replace(".jsonl", "_summary.json")
    dims = sorted({r.get("vector_dim", 0) for r in records})

    summary = {
        "count": len(records),
        "models": sorted({r.get("model") for r in records}),
        "vector_dims": dims,
        "output": output_path,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary_path


def build_cli():
    parser = argparse.ArgumentParser(
        description="Generate embeddings from HTML chunk JSON using OpenAI text-embedding-3-large"
    )
    parser.add_argument(
        "--input",
        default="data/raw_html/ecfr/section_668_32_chunks.json",
        help="Input chunk JSON path",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output embeddings JSONL path (default: <input>_embeddings.jsonl)",
    )
    parser.add_argument(
        "--chunk-source",
        choices=["base_chunks", "level_chunks"],
        default="base_chunks",
        help="Which chunk array to embed",
    )
    parser.add_argument(
        "--model",
        default=MODEL,
        help="Embedding model",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Embedding batch size",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=4,
        help="Max retries per batch",
    )
    parser.add_argument(
        "--retry-sleep",
        type=float,
        default=1.5,
        help="Retry backoff base seconds",
    )
    return parser


def main():
    args = build_cli().parse_args()

    output_path = args.output.strip() or default_output_path(args.input)
    chunks = load_chunks(args.input, chunk_source=args.chunk_source)

    records = create_embeddings(
        chunks=chunks,
        model=args.model,
        batch_size=args.batch_size,
        max_retries=args.max_retries,
        retry_sleep=args.retry_sleep,
    )

    write_jsonl(records, output_path)
    summary_path = write_summary(records, output_path)

    print(f"Saved embeddings: {output_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
