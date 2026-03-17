# Hybrid Search POC (Elasticsearch + Qdrant)

A small pipeline that ingests PDFs from a directory, extracts text, chunks it, embeds with OpenAI, and writes to **Elasticsearch** (keyword search) and **Qdrant** (vector search) in parallel. All configuration is driven by `.env`. Successful PDFs are moved to a `processed` directory.

---

## 1. Functionality

**End-to-end flow**

1. **Discovery** — Scans `DATA_DIR` (e.g. `./data`) for `*.pdf` files.
2. **Extraction** — Reads each PDF with PyPDF and concatenates text from all pages.
3. **Chunking** — Splits text into overlapping segments of `CHUNK_SIZE` characters with `CHUNK_OVERLAP` overlap. Each chunk gets a stable id like `filename.pdf_0`, `filename.pdf_1`, etc.
4. **Embedding** — Sends chunks to the OpenAI embeddings API in batches of `EMBED_BATCH_SIZE` (to stay under the 300k-tokens-per-request limit). Uses the model set in `OPENAI_EMBEDDING_MODEL` (e.g. `text-embedding-3-small`).
5. **Dual write** — For each chunk, writes to **Elasticsearch** (document id, `text`, `source`) and **Qdrant** (point id, vector, payload with `text`, `source`, `chunk_id`) in parallel via a thread pool.
6. **Processed move** — If every chunk was written successfully, moves each ingested PDF from `DATA_DIR` to `DATA_PROCESSED_DIR` (e.g. `data/processed`). If any write failed, no files are moved so you can fix and re-run.

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
