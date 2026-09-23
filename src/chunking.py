"""
Step 2: Chunk (Silver layer)

Implements two chunking strategies so we can compare which one gives better
retrieval quality later:

1. Fixed-size chunking  — split every N characters with some overlap.
   Simple, predictable, but can cut sentences/ideas in half.

2. Paragraph-based ("semantic") chunking — split on natural paragraph/section
   boundaries, then merge tiny paragraphs together so chunks aren't too small.
   Keeps ideas intact, but chunk sizes vary.

Each chunk is saved with metadata: source file, chunk index, and strategy —
this metadata is what lets us filter/compare in Pinecone later.
"""
import os
import json
import glob

import config


def fixed_size_chunks(text: str, chunk_size: int, overlap: int):
    """Slide a fixed-size window over the text with the given overlap."""
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_len:
            break
        start = end - overlap
    return chunks


def paragraph_chunks(text: str, min_chunk_chars: int = 300, max_chunk_chars: int = 1200):
    """
    Split on blank-line paragraph boundaries, then greedily merge consecutive
    paragraphs until a chunk reaches min_chunk_chars, capping at max_chunk_chars.
    """
    raw_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    if len(raw_paragraphs) <= 1:
        raw_paragraphs = [line.strip() for line in text.split("\n") if line.strip()]

    chunks = []
    current = ""
    for para in raw_paragraphs:
        candidate = (current + "\n\n" + para).strip() if current else para
        if len(candidate) > max_chunk_chars and current:
            chunks.append(current)
            current = para
        else:
            current = candidate
            if len(current) >= min_chunk_chars:
                chunks.append(current)
                current = ""
    if current:
        chunks.append(current)

    return chunks


def chunk_file(filepath: str, strategy: str):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    if strategy == "fixed":
        pieces = fixed_size_chunks(text, config.FIXED_CHUNK_SIZE, config.FIXED_CHUNK_OVERLAP)
    elif strategy == "semantic":
        pieces = paragraph_chunks(text)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    source_name = os.path.basename(filepath)
    records = [
        {
            "id": f"{source_name}::{strategy}::{i}",
            "text": piece,
            "source": source_name,
            "strategy": strategy,
            "chunk_index": i,
        }
        for i, piece in enumerate(pieces)
    ]
    return records


def chunk_all(raw_dir=None, out_dir=None):
    raw_dir = raw_dir or config.RAW_DIR
    out_dir = out_dir or config.CHUNKS_DIR
    os.makedirs(out_dir, exist_ok=True)

    all_files = glob.glob(os.path.join(raw_dir, "*.txt"))
    if not all_files:
        print(f"No .txt files found in {raw_dir}. Run scraper.py first.")
        return

    for strategy in ("fixed", "semantic"):
        all_records = []
        for filepath in all_files:
            all_records.extend(chunk_file(filepath, strategy))

        out_path = os.path.join(out_dir, f"chunks_{strategy}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_records, f, indent=2)

        sizes = [len(r["text"]) for r in all_records]
        avg_size = sum(sizes) / len(sizes) if sizes else 0
        print(
            f"[{strategy}] {len(all_records)} chunks from {len(all_files)} files "
            f"| avg size: {avg_size:.0f} chars | saved to {out_path}"
        )


if __name__ == "__main__":
    chunk_all()