"""
Step 3: Embed + Store (Gold layer)

Loads the chunked JSON files, generates embeddings with a free local model
(sentence-transformers), and upserts them into two separate Pinecone
namespaces — one per chunking strategy.

Run this with PINECONE_API_KEY set:
    python src/embed_and_store.py
"""
import os
import json

from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

import config


def get_pinecone_index():
    """Connect to Pinecone, creating the index if it doesn't exist yet."""
    if not config.PINECONE_API_KEY:
        raise RuntimeError(
            "PINECONE_API_KEY is not set. Get a free key at https://www.pinecone.io/ "
            "and set it in your terminal before running this script."
        )

    pc = Pinecone(api_key=config.PINECONE_API_KEY)

    existing = [idx["name"] for idx in pc.list_indexes()]
    if config.PINECONE_INDEX_NAME not in existing:
        print(f"Creating Pinecone index '{config.PINECONE_INDEX_NAME}'...")
        pc.create_index(
            name=config.PINECONE_INDEX_NAME,
            dimension=config.EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=config.PINECONE_CLOUD, region=config.PINECONE_REGION),
        )

    return pc.Index(config.PINECONE_INDEX_NAME)


def load_chunks(strategy: str):
    path = os.path.join(config.CHUNKS_DIR, f"chunks_{strategy}.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def embed_and_upsert(strategy: str, namespace: str, model, index, batch_size: int = 50):
    records = load_chunks(strategy)
    if not records:
        print(f"No chunks found for strategy '{strategy}'. Run chunking.py first.")
        return

    print(f"Embedding {len(records)} '{strategy}' chunks...")
    texts = [r["text"] for r in records]
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    vectors = [
        {
            "id": records[i]["id"],
            "values": embeddings[i].tolist(),
            "metadata": {
                "text": records[i]["text"][:2000],
                "source": records[i]["source"],
                "strategy": records[i]["strategy"],
                "chunk_index": records[i]["chunk_index"],
            },
        }
        for i in range(len(records))
    ]

    for i in range(0, len(vectors), batch_size):
        batch = vectors[i : i + batch_size]
        index.upsert(vectors=batch, namespace=namespace)
        print(f"    upserted {i + len(batch)}/{len(vectors)} vectors to namespace '{namespace}'")


def main():
    print(f"Loading embedding model '{config.EMBEDDING_MODEL_NAME}' (downloads once, then cached)...")
    model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)

    index = get_pinecone_index()

    embed_and_upsert("fixed", config.NAMESPACE_FIXED, model, index)
    embed_and_upsert("semantic", config.NAMESPACE_SEMANTIC, model, index)

    print("\nDone. Both namespaces are populated — ready to query with query.py")


if __name__ == "__main__":
    main()