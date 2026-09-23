# RAG Pipeline with Chunking Strategy Comparison

A small end-to-end RAG (Retrieval-Augmented Generation) data pipeline that scrapes
Apache Spark documentation, chunks it two different ways, embeds it, and stores it
in Pinecone — so you can directly compare which chunking strategy retrieves better
context for a given question.

## Why this project

Data engineering for AI increasingly means building the pipelines that feed LLM
applications, not just BI dashboards. This project covers the full RAG data path:

```
Scrape (Bronze) -> Clean & Chunk (Silver) -> Embed & Store (Gold) -> Query & Compare
```

It also demonstrates a real engineering decision — chunking strategy — instead of
just wiring tools together. That comparison is the part worth talking about in an
interview.

## Chunking strategies compared

1. **Fixed-size chunking** — splits text every 500 characters with 50 characters
   of overlap. Simple and predictable, but can cut a sentence or idea in half.
2. **Paragraph-based ("semantic") chunking** — splits on natural paragraph
   boundaries and merges small paragraphs together until each chunk is a
   reasonable size. Keeps ideas intact, but chunk sizes vary more.

Both are stored in separate Pinecone namespaces (`fixed-chunking` and
`semantic-chunking`) inside the same index, so a single query script can retrieve
from both and show you the difference side by side.

## Setup

Run this on your own machine (not a sandboxed environment) since step 1 needs
real internet access.

```bash
# 1. Create a virtual environment
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Get a free Pinecone API key at https://www.pinecone.io/
#    Then copy .env.example to .env and fill in your key, or export it directly:
export PINECONE_API_KEY=your_key_here
```

## Running the pipeline

```bash
cd src

# Step 1: Scrape the Spark doc pages listed in config.py
python scraper.py

# Step 2: Chunk the scraped text two ways
python chunking.py

# Step 3: Embed chunks (local model, free) and upload to Pinecone
python embed_and_store.py

# Step 4: Ask a question and compare retrieval quality across both strategies
python query.py "How does Structured Streaming handle late-arriving data?"
```

## What to put in your portfolio write-up

After running a handful of queries, note down:
- Which strategy's top-ranked chunk actually answered the question more completely
- Whether fixed-size chunking ever cut off a key sentence mid-thought
- Whether paragraph-based chunking ever returned a chunk that was too long/unfocused
- The average chunk count and size for each strategy (printed by `chunking.py`)

That comparison — backed by real retrieved examples — is what makes this project
more than a tutorial clone.

## Extending this further

- Swap the local embedding model for an API-based one (OpenAI/Groq) and compare cost/quality
- Add an LLM call (e.g. Groq, which you're already using in Smart Finance Mentor) that
  takes the retrieved chunks and generates a final answer, completing the "G" in RAG
- Add a third chunking strategy (e.g. sentence-window or recursive character splitting
  via LangChain) for a three-way comparison
- Wrap `query.py` in a small Streamlit app so it's demoable, not just a CLI script

## Project structure

```
rag-pipeline/
├── src/
│   ├── config.py           # URLs, API keys, chunking settings
│   ├── scraper.py          # Step 1: scrape + clean HTML
│   ├── chunking.py         # Step 2: two chunking strategies
│   ├── embed_and_store.py  # Step 3: embed + upsert to Pinecone
│   └── query.py            # Step 4: query + compare both strategies
├── data/
│   ├── raw/                # Scraped text, one file per page (Bronze)
│   └── chunks/             # Chunked JSON, one file per strategy (Silver)
├── requirements.txt
├── .env.example
└── README.md
```
