"""
Step 4: Query + Compare

Takes a natural-language question, embeds it with the same local model used
for the chunks, and queries BOTH Pinecone namespaces so you can compare
retrieval quality side by side.

Usage:
    python src/query.py "How does Structured Streaming handle late data?"
"""
import sys

from sentence_transformers import SentenceTransformer

import config
from embed_and_store import get_pinecone_index


def query_namespace(index, model, question: str, namespace: str, top_k: int = 3):
    query_vector = model.encode([question], normalize_embeddings=True)[0].tolist()
    result = index.query(
        vector=query_vector,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True,
    )
    return result.get("matches", [])


def print_matches(strategy_label: str, matches):
    print(f"\n{'=' * 20} {strategy_label} {'=' * 20}")
    if not matches:
        print("(no matches — did you run embed_and_store.py?)")
        return
    for rank, match in enumerate(matches, start=1):
        meta = match.get("metadata", {})
        score = match.get("score", 0)
        text_preview = meta.get("text", "")[:300].replace("\n", " ")
        print(f"\n#{rank} | score={score:.4f} | source={meta.get('source')} | chunk #{meta.get('chunk_index')}")
        print(f"    {text_preview}...")


def main():
    if len(sys.argv) < 2:
        print('Usage: python src/query.py "your question here"')
        sys.exit(1)

    question = sys.argv[1]
    print(f"Question: {question}")

    model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    index = get_pinecone_index()

    fixed_matches = query_namespace(index, model, question, config.NAMESPACE_FIXED)
    semantic_matches = query_namespace(index, model, question, config.NAMESPACE_SEMANTIC)

    print_matches("FIXED-SIZE CHUNKING", fixed_matches)
    print_matches("PARAGRAPH-BASED CHUNKING", semantic_matches)


if __name__ == "__main__":
    main()