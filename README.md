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

## Findings

I tested both chunking strategies against Pinecone using a mix of specific,
combinatorial, and broad questions about Spark's documentation. Three real
results are worth documenting:

### 1. A scraping bug taught me not to trust results blindly

My first test run scraped `structured-streaming-programming-guide.html`
directly — but Spark 4.0 replaced that page with a short redirect notice
pointing to a *new* set of split-up pages. My scraper faithfully extracted
that redirect stub (186 characters, no real content), and both chunking
strategies returned low-relevance results for every Structured Streaming
question — not because of a chunking problem, but because the actual source
content was never captured in the first place.

I fixed this by tracing the real destination URLs (`streaming/index.html`
and its sub-pages) and updating `SOURCE_URLS` in `config.py`, plus added a
safety check in `scraper.py` that now warns immediately if a scraped page
comes back under 500 characters — instead of the problem surfacing three
steps downstream at query time.

**Takeaway:** a RAG pipeline is only as good as its source coverage. Bad
retrieval scores can be a chunking problem, an embedding problem, or —
as it was here — a data-completeness problem further upstream. Always
verify the raw scraped content before trusting the numbers on top of it.

### 2. A higher similarity score doesn't always mean a more relevant chunk

For the question *"How does Structured Streaming handle late-arriving data?"*,
after fixing the scraping bug:

| Strategy | Top score | Actually about late data? |
|---|---|---|
| Fixed-size | 0.688 | Yes — explains the 2-hour late-data threshold directly |
| Paragraph-based | 0.795 (highest of either strategy) | No — about asynchronous progress tracking/checkpointing |

Paragraph-based chunking's top result scored *higher* but was topically
off — it matched on shared vocabulary ("processing," "latency") rather
than the actual concept asked about. Fixed-size chunking's top-2 results
were both directly on-topic despite lower raw scores.

This pattern repeated on an unrelated question about watermarking, where a
"Stream-Static join support" compatibility table — not a watermarking
document at all — ranked #1 for fixed-size and #2 for semantic chunking,
pulled in purely by shared vocabulary ("streaming," "stateful,"
"supported").

**Takeaway:** similarity score measures embedding closeness, not
correctness. Evaluating a RAG system means reading what was actually
retrieved, not just trusting the ranking metric — a lesson that matters
more than which strategy "wins."

### 3. A chunking bug produced context-free orphan fragments — found, fixed, and verified

For the question *"How does Structured Streaming achieve exactly-once
semantics?"*, paragraph-based chunking's top result was:

> `"end-to-end exactly-once semantics under any failure...."`

A single sentence fragment with no surrounding context — not useful on its
own, despite scoring highest (0.6518). Meanwhile, the chunk with the
complete, self-contained "At least once" / "Exactly once" definitions
scored lowest of the three shown (0.5175) and ranked #3.

The root cause was a bug in `paragraph_chunks()`: when the loop over a
source page finished, any small leftover text (`current`) was appended as
its own chunk regardless of size — so a page ending mid-thought produced a
tiny, context-free chunk with an inflated relevance score, purely because
short chunks tend to match narrowly and score high on specific queries.

**The fix**, in `chunking.py`:

```python
if current:
    if chunks and len(current) < min_chunk_chars:
        # Too small to stand alone — merge into the previous chunk
        # instead of leaving an orphaned fragment with no context.
        chunks[-1] = chunks[-1] + "\n\n" + current
    else:
        chunks.append(current)
```

**Before vs. after re-running the pipeline**, same question, paragraph-based
chunking:

| | Rank | Score | Content |
|---|---|---|---|
| Before | #1 | 0.6518 | Orphan fragment — no context |
| Before | #3 | 0.5175 | Full definitions of all three guarantee types |
| After | #1 | 0.7320 | Full definitions of all three guarantee types |
| After | — | — | Orphan fragment no longer appears in top 3 |

The genuinely useful chunk moved from #3 to #1, and its score rose because
merging gave it more surrounding context and keyword density.

Interestingly, fixed-size chunking's top result for the same question
started mid-word (`"gether, using replayable sources and idempotent
sinks..."` — the tail end of "altogether"). This looks similar to the
orphan problem but is actually a different, more fundamental issue: fixed-
size chunking cuts at a fixed character count regardless of where a word or
sentence falls, so this can't be patched the way the paragraph-chunker's
bug was — it's inherent to the strategy.

**Takeaway:** not every chunking flaw is inherent to the strategy — some are
implementation bugs. The orphan-fragment issue was fixable with a small,
targeted patch (and worth catching before it silently deflates a chunking
strategy's evaluation). Fixed-size chunking's mid-word cutoff, by contrast,
is a structural limitation of the approach itself.

### Overall

Across this dataset, neither strategy dominated consistently — results
were close and sometimes contradicted the score ranking, as above. If
anything, fixed-size chunking's predictable boundaries made it slightly
easier to reason about for these documentation-style questions, while
paragraph-based chunking's variable size sometimes produced longer,
less focused chunks — or, before the fix above, the occasional
context-free orphan — that pulled in tangential content.

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
  or, before the fix in Finding 3, a context-free orphan fragment
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
- Extend the orphan-merge fix to catch mid-document fragments, not just ones at the
  end of a page's chunk list

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