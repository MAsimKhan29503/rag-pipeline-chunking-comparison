"""
Central configuration for the RAG pipeline.
API keys are read from environment variables — never hardcode them.
"""
import os

# --- Pinecone ---
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "spark-docs-rag")
PINECONE_CLOUD = os.environ.get("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.environ.get("PINECONE_REGION", "us-east-1")  # free tier region

# --- Embedding model (local, free) ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # 384-dim, fast, good quality for this scale
EMBEDDING_DIM = 384

# --- Source documents to scrape ---
SOURCE_URLS = [
    "https://spark.apache.org/docs/latest/streaming/index.html",
    "https://spark.apache.org/docs/latest/streaming/getting-started.html",
    "https://spark.apache.org/docs/latest/streaming/apis-on-dataframes-and-datasets.html",
    "https://spark.apache.org/docs/latest/streaming/performance-tips.html",
    "https://spark.apache.org/docs/latest/streaming/additional-information.html",
    "https://spark.apache.org/docs/latest/sql-programming-guide.html",
    "https://spark.apache.org/docs/latest/rdd-programming-guide.html",
    "https://spark.apache.org/docs/latest/streaming-programming-guide.html",
]

# --- Chunking ---
FIXED_CHUNK_SIZE = 500       # characters
FIXED_CHUNK_OVERLAP = 50     # characters

# --- Namespaces in Pinecone (lets us compare the two strategies side by side) ---
NAMESPACE_FIXED = "fixed-chunking"
NAMESPACE_SEMANTIC = "semantic-chunking"

# --- Paths ---
RAW_DIR = "data/raw"
CHUNKS_DIR = "data/chunks"