# Hybrid Search POC (Elasticsearch + Qdrant)

This project has two parts:

- **Ingestion**: ingest PDFs, extract + chunk text, embed with OpenAI, then write chunks to **Elasticsearch** (keyword/BM25) and **Qdrant** (vector) in parallel.
- **Hybrid search + UI**: run **parallel retrieval** against Elasticsearch + Qdrant, **normalize** scores, **fuse** them with weights, and **rank** final results. A Streamlit chat UI uses the top chunks as context to produce a grounded answer via OpenAI.

All configuration is driven by `.env`. Successful PDFs are moved to a `processed` directory after a fully successful ingest.

---

## 1. Functionality

### End-to-end flow

#### A) Ingestion (PDFs → ES + Qdrant)

1. **Discovery** — Scans `DATA_DIR` (e.g. `./data`) for `*.pdf` files.
2. **Extraction** — Reads each PDF with PyPDF and concatenates text from all pages.
3. **Chunking** — Splits text into overlapping segments of `CHUNK_SIZE` characters with `CHUNK_OVERLAP` overlap. Each chunk gets a stable id like `filename.pdf_0`, `filename.pdf_1`, etc.
4. **Embedding** — Sends chunks to the OpenAI embeddings API in batches of `EMBED_BATCH_SIZE`. Uses `OPENAI_EMBEDDING_MODEL` (e.g. `text-embedding-3-small`).
5. **Dual write** — For each chunk, writes to **Elasticsearch** (`text`, `source`) and **Qdrant** (vector + payload with `text`, `source`, `chunk_id`) in parallel.
6. **Processed move** — If every chunk is written successfully, moves each ingested PDF from `DATA_DIR` to `DATA_PROCESSED_DIR`. If any write fails, no files are moved.

#### B) Hybrid search (ES + Qdrant → fused ranking)

When you query in Streamlit (or call `hybrid_search` directly):

1. **Parallel retrieval**:
   - Elasticsearch keyword/BM25 search over `text` in `ES_INDEX_NAME`
   - Qdrant vector search over `QDRANT_COLLECTION_NAME` using an OpenAI-embedded query
2. **Normalize scores**: min-max normalize each result list independently to \([0, 1]\).
3. **Build lookups**: align candidates by `chunk_id`, keeping `text` and `source`.
4. **Weighted fusion**: compute \(hybrid = w_{bm25} \cdot bm25_{norm} + w_{vec} \cdot vec_{norm}\).
5. **Final ranking**: sort by `hybrid_score` and return the top `HYBRID_RETURN_K`.

**Output**

- **Elasticsearch**: Index `ES_INDEX_NAME` with fields `text` (full-text) and `source` (keyword). Document IDs are chunk ids (e.g. `doc.pdf_0`).
- **Qdrant**: Collection `QDRANT_COLLECTION_NAME` with vectors of dimension `OPENAI_EMBEDDING_DIM`. Point IDs are integers; payload holds `text`, `source`, and `chunk_id` for correlation with Elasticsearch.

**Logging**

- Per file: “Processing: …”, “Completed: … (N chunks)” or “Skipped …”.
- Summary: total chunks, files, “Index/collection ready”, progress every 50 chunks.
- Final: total chunks, success/failed counts, status OK or COMPLETED WITH ERRORS, and “Moved to processed: …” for each file when the run is fully successful.

---

## 2. Configuration

All settings come from environment variables. Copy `.env.example` to `.env` and set values as needed.

**Required**

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key (required for embeddings). |

### Tunable knobs (what you can tweak)

This project has three main tuning surfaces:

- **Ingest/chunking**: how PDFs are split into chunks before embedding.
- **Retrieval**: how many candidates to pull from each store and how to fuse them.
- **Generation**: which chat model to use for the Streamlit answer.

**Store URLs**

| Variable | Default | Description |
|----------|---------|-------------|
| `ELASTICSEARCH_URL` | `http://localhost:9200` | Elasticsearch base URL. |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant base URL. |

**Ingest paths and names**

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_DIR` | `./data` | Directory to scan for PDFs. |
| `DATA_PROCESSED_DIR` | `./data/processed` | Directory where successfully ingested PDFs are moved. |
| `ES_INDEX_NAME` | `hybrid-docs` | Elasticsearch index name. |
| `QDRANT_COLLECTION_NAME` | `hybrid-docs` | Qdrant collection name. |

**OpenAI embedding**

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model name. |
| `OPENAI_EMBEDDING_DIM` | `1536` | Vector size (must match the model; 1536 for `text-embedding-3-small`). |
| `EMBED_BATCH_SIZE` | `1000` | Max chunks per embedding API request (OpenAI limit 300k tokens/request). |

**Chunking**

| Variable | Default | Description |
|----------|---------|-------------|
| `CHUNK_SIZE` | `500` | Chunk length in characters. |
| `CHUNK_OVERLAP` | `50` | Overlap between consecutive chunks in characters. |

### Hybrid retrieval tuning

These affect `src/search.py` (`hybrid_search`) when callers don’t pass explicit values.

| Variable | Default | Description |
|----------|---------|-------------|
| `HYBRID_TOP_K` | `50` | How many candidates to fetch from **each** retriever (ES + Qdrant) before fusion/normalization. |
| `HYBRID_RETURN_K` | `20` | How many results to keep **after** fusion + final ranking (returned to the app). |
| `HYBRID_BM25_WEIGHT` | `0.4` | Weight for normalized Elasticsearch BM25 scores. |
| `HYBRID_VECTOR_WEIGHT` | *(blank)* | Weight for normalized Qdrant vector scores. If blank, defaults to \(1 -\) `HYBRID_BM25_WEIGHT`. |

**Rule of thumb**

- Set `HYBRID_TOP_K` **>=** `HYBRID_RETURN_K` (often 2–5× larger) so fusion has enough candidates.
- If you mostly want **exact keyword matches**, increase `HYBRID_BM25_WEIGHT`.
- If you mostly want **semantic similarity**, decrease `HYBRID_BM25_WEIGHT` (or set `HYBRID_VECTOR_WEIGHT` explicitly).

---

## 3. Running the ingestion

**Prerequisites**

- Docker (for Elasticsearch and Qdrant).
- Python 3.10+.
- OpenAI API key in `.env`.

**Setup**

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set OPENAI_API_KEY (and any other overrides).
```

**Start the stack**

```bash
docker compose up -d
```

Elasticsearch will be on port 9200, Qdrant on 6333. Data is stored in Docker volumes (`esdata`, `qdrant_storage`) so it persists across restarts.

**Add PDFs**

Place PDFs in the directory configured as `DATA_DIR` (default `./data`). Only `*.pdf` files are processed.

**Run the ingestor**

```bash
python -m src.main
```

Or:

```bash
python -m src.ingest
```

**What to expect**

- Logs for each file: “Processing: …”, then “Completed: … (N chunks)” or a skip/warning.
- “Loaded N chunks from M file(s)”, then “Index/collection ready. Embedding and writing to Elasticsearch + Qdrant…”.
- “Progress: X/Y” every 50 chunks and at the end.
- Final block: “--- Ingest complete ---”, total chunks, success/failed, status. If all succeeded, “Moved to processed: …” for each PDF.
- Processed PDFs will be under `DATA_PROCESSED_DIR` (e.g. `data/processed/`); they are removed from `DATA_DIR`.

**Re-runs**

- Put new PDFs in `DATA_DIR` and run again. Already-processed PDFs in `data/processed` are not re-ingested.
- To re-ingest everything, move PDFs back from `data/processed` to `data/` and run. Note: the pipeline does not delete or replace existing index/collection data; you may get duplicates unless you clear the index/collection or use a new index/collection name in `.env`.

**Tests**

```bash
pytest
```

The connectivity test pings Elasticsearch and Qdrant using `.env` URLs; it is skipped if the services are not reachable (e.g. Docker not running).

---

## 4. Running the Streamlit app (Hybrid Search + Chat)

This UI performs **hybrid retrieval** (Elasticsearch BM25 + Qdrant vector search in parallel), normalizes scores, fuses them with weights, and then generates a grounded answer using OpenAI.

### Detailed functionality (UI path)

When you ask a question in the Streamlit chat:

1. **Parallel retrieval**:
   - Elasticsearch keyword search over `text` in `ES_INDEX_NAME`
   - Qdrant vector search over `QDRANT_COLLECTION_NAME` using an OpenAI-embedded query
2. **Normalize scores**: min-max normalize each result list independently to \([0, 1]\).
3. **Build lookups**: align candidates by `chunk_id`, keeping `text` and `source`.
4. **Weighted fusion**: compute \(hybrid = w_{bm25} \cdot bm25_{norm} + w_{vec} \cdot vec_{norm}\).
5. **Final ranking**: sort by `hybrid_score` descending and return `HYBRID_RETURN_K` items.
6. **Grounded answer generation**: send the top few chunks as context to the chat model and instruct it to answer **only** from context.

### Tunable config (generation)

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | Chat model used for answer generation. |

**Run**

```bash
# from this project folder
docker compose up -d
source .venv/bin/activate
streamlit run src/app.py
```

**Notes**

- The retrieval uses `ES_INDEX_NAME` / `QDRANT_COLLECTION_NAME` from `.env` (defaults `hybrid-docs`). Run ingestion first so there is data to search.
- The Streamlit sidebar controls the BM25 vs vector fusion weight; those slider values override `.env` weights for that session.

---

## 5. Evaluation Dashboard (quality + baselines)

The Streamlit app includes an **Evaluation** tab that helps you **assess and document** search/RAG quality over time.

### What it logs (JSONL)

Each chat run is appended to `EVAL_LOG_PATH` as one JSON line with:

- `query`
- `config` (fusion method, weights, top_k/return_k, RRF k, etc.)
- `timings_ms` (ES, Qdrant, fusion, total)
- `retrieval` (ranked chunks with scores)
- `answer`
- `judge` (optional; filled when you run the judge)

### What the judge measures (no-label eval)

The dashboard uses an OpenAI model as a strict evaluator (“LLM-as-judge”) to score:

- **groundedness (1–5)**: answer supported by retrieved context only
- **citation_precision (1–5)**: retrieved chunks support claims made
- **helpfulness (1–5)** and **completeness (1–5)** relative to context
- **refusal_correctness (1–5)** when context lacks an answer
- **hallucination_flag (bool)** for unsupported claims

### Ragas scoring (optional)

If you install [Ragas](https://github.com/vibrantlabsai/ragas) (`pip install ragas`), the Evaluation tab can use it instead of the custom LLM judge. In **Controls** choose **Scoring backend: Ragas**. Ragas computes:

- **faithfulness** (0–1): factual consistency of the answer with retrieved context
- **answer_relevancy** (0–1): how relevant the answer is to the question

The dashboard maps these to the same KPI labels (groundedness, helpfulness, etc.) so charts and tables work unchanged.

### Baseline comparison: Weighted vs RRF

To compare initial rankings on your dataset, you can A/B test:

- `HYBRID_FUSION_METHOD=weighted_minmax` (min-max normalize then weighted sum)
- `HYBRID_FUSION_METHOD=rrf` (Reciprocal Rank Fusion baseline; `HYBRID_RRF_K` default 60)

The Evaluation tab can run A/B evaluations over the **last N queries** and append results to the log for later analysis.

### Eval-related configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `EVAL_LOG_PATH` | `./eval_runs.jsonl` | Where Streamlit appends run records (JSONL). |
| `HYBRID_FUSION_METHOD` | `weighted_minmax` | Fusion method used by default (`weighted_minmax` or `rrf`). |
| `HYBRID_RRF_K` | `60` | RRF stabilizer constant. |
| `OPENAI_JUDGE_MODEL` | `gpt-4o-mini` | Model used for LLM-as-judge scoring. |
