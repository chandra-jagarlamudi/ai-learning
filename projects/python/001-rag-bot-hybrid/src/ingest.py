"""
Ingest PDFs from DATA_DIR: extract text, chunk, embed with OpenAI, write to Elasticsearch and Qdrant in parallel.
All configuration via .env.
"""
import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from openai import OpenAI
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

load_dotenv()

# Config from env
DATA_DIR = os.getenv("DATA_DIR", "./data")
DATA_PROCESSED_DIR = os.getenv("DATA_PROCESSED_DIR", "./data/processed")
ES_INDEX_NAME = os.getenv("ES_INDEX_NAME", "hybrid-docs")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "hybrid-docs")
ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
# OpenAI max 300k tokens/request; batch size in number of chunks per API call
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "1000"))
# Embedding vector size (must match model; text-embedding-3-small = 1536)
OPENAI_EMBEDDING_DIM = int(os.getenv("OPENAI_EMBEDDING_DIM", "1536"))
# Text chunking: size and overlap in characters
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))


def _chunk_text(text: str, source: str) -> list[tuple[str, str, str]]:
    """Split text into overlapping chunks. Returns [(chunk_id, text, source), ...]."""
    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        piece = text[start:end].strip()
        if piece:
            chunk_id = f"{source}_{idx}"
            chunks.append((chunk_id, piece, source))
            idx += 1
        start = end - CHUNK_OVERLAP
        if start >= len(text):
            break
    return chunks


def _extract_pdf_text(path: Path) -> str:
    reader = PdfReader(path)
    return "\n".join(p.extract_text() or "" for p in reader.pages)


def _load_documents(data_dir: str) -> list[tuple[str, str, str]]:
    """Discover PDFs in data_dir, extract text, chunk. Returns [(chunk_id, text, source), ...]."""
    root = Path(data_dir)
    if not root.is_dir():
        return []
    all_chunks = []
    paths = sorted(root.glob("*.pdf"))
    for path in paths:
        try:
            log.info("Processing: %s", path.name)
            text = _extract_pdf_text(path)
            if text.strip():
                chunks = _chunk_text(text, path.name)
                for item in chunks:
                    all_chunks.append(item)
                log.info("  Completed: %s (%d chunks)", path.name, len(chunks))
            else:
                log.warning("  Skipped (no text): %s", path.name)
        except Exception as e:
            log.warning("  Skip %s: %s", path.name, e)
    return all_chunks


def _embed(texts: list[str], client: OpenAI) -> list[list[float]]:
    """OpenAI embeddings for a list of texts. Batches to stay under 300k tokens/request."""
    if not texts:
        return []
    out = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        r = client.embeddings.create(model=OPENAI_EMBEDDING_MODEL, input=batch)
        out.extend(item.embedding for item in r.data)
    return out


def _ensure_es_index(es: Elasticsearch) -> None:
    try:
        es.indices.create(
            index=ES_INDEX_NAME,
            mappings={
                "properties": {
                    "text": {"type": "text"},
                    "source": {"type": "keyword"},
                }
            },
        )
    except Exception as e:
        body = getattr(e, "body", None) or {}
        err = body.get("error", {}) if isinstance(body, dict) else {}
        err_type = err.get("type", "") if isinstance(err, dict) else ""
        if err_type != "resource_already_exists_exception" and "resource_already_exists_exception" not in str(e):
            raise


def _ensure_qdrant_collection(qdrant: QdrantClient) -> None:
    collections = qdrant.get_collections().collections
    names = [c.name for c in collections]
    if QDRANT_COLLECTION_NAME not in names:
        qdrant.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(size=OPENAI_EMBEDDING_DIM, distance=Distance.COSINE),
        )


def _write_es(es: Elasticsearch, chunk_id: str, text: str, source: str) -> None:
    es.index(
        index=ES_INDEX_NAME,
        id=chunk_id,
        document={"text": text, "source": source},
    )


def _write_qdrant(
    qdrant: QdrantClient, point_id: int, vector: list[float], text: str, source: str, chunk_id: str
) -> None:
    """Qdrant accepts only int or UUID for point id; we use index and keep chunk_id in payload."""
    point = PointStruct(
        id=point_id,
        vector=vector,
        payload={"text": text, "source": source, "chunk_id": chunk_id},
    )
    qdrant.upsert(collection_name=QDRANT_COLLECTION_NAME, points=[point])


def run() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        log.error("OPENAI_API_KEY not set. Set it in .env")
        return

    log.info("Starting ingestion from DATA_DIR=%s", DATA_DIR)
    documents = _load_documents(DATA_DIR)
    if not documents:
        log.warning("No PDFs or chunks in %s. Add PDFs and run again.", DATA_DIR)
        return

    n_files = len({c[2] for c in documents})
    log.info("Loaded %d chunks from %d file(s)", len(documents), n_files)
    log.info("Index/collection ready. Embedding and writing to Elasticsearch + Qdrant...")

    # Force compatibility with ES 8 server (client may send 9 otherwise)
    es = Elasticsearch(
        ELASTICSEARCH_URL,
        headers={"Accept": "application/vnd.elasticsearch+json; compatible-with=8"},
    )
    qdrant = QdrantClient(url=QDRANT_URL)
    openai_client = OpenAI()

    _ensure_es_index(es)
    _ensure_qdrant_collection(qdrant)

    chunk_ids = [c[0] for c in documents]
    texts = [c[1] for c in documents]
    sources = [c[2] for c in documents]
    vectors = _embed(texts, openai_client)

    success = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=2) as executor:
        for i, (cid, text, source, vec) in enumerate(zip(chunk_ids, texts, sources, vectors)):
            f_es = executor.submit(_write_es, es, cid, text, source)
            f_qd = executor.submit(_write_qdrant, qdrant, i, vec, text, source, cid)
            try:
                f_es.result()
                f_qd.result()
                success += 1
            except Exception as e:
                failed += 1
                log.warning("Error for %s: %s", cid, e)
            if (i + 1) % 50 == 0 or (i + 1) == len(documents):
                log.info("  Progress: %d/%d", i + 1, len(documents))

    log.info("--- Ingest complete ---")
    log.info("  Total chunks: %d", len(documents))
    log.info("  Success: %d", success)
    log.info("  Failed: %d", failed)
    log.info("  Status: %s", "OK" if failed == 0 else "COMPLETED WITH ERRORS")

    if failed == 0 and documents:
        processed_dir = Path(DATA_PROCESSED_DIR)
        processed_dir.mkdir(parents=True, exist_ok=True)
        for source in {c[2] for c in documents}:
            src_path = Path(DATA_DIR) / source
            if src_path.is_file():
                dst_path = processed_dir / source
                shutil.move(str(src_path), str(dst_path))
                log.info("  Moved to processed: %s", source)


if __name__ == "__main__":
    run()
