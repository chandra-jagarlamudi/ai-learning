import os

import pytest
from dotenv import load_dotenv

load_dotenv()


def test_smoke():
    assert 1 + 1 == 2


def test_connectivity_es_and_qdrant():
    """With docker compose up, ES and Qdrant should be reachable (uses .env URLs). Skips if unreachable."""
    es_url = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")

    from elasticsearch import Elasticsearch
    from qdrant_client import QdrantClient

    try:
        es = Elasticsearch(es_url)
        info = es.info()
        assert "version" in info
    except Exception as e:
        pytest.skip(f"Elasticsearch not reachable at {es_url}: {e}")

    try:
        qdrant = QdrantClient(url=qdrant_url)
        collections = qdrant.get_collections()
        assert collections is not None
    except Exception as e:
        pytest.skip(f"Qdrant not reachable at {qdrant_url}: {e}")
