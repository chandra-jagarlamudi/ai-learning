import asyncio
import os
import time
from typing import Dict, List, Optional, Tuple, Union

import logging
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from openai import OpenAI
from qdrant_client import QdrantClient

load_dotenv()

log = logging.getLogger(__name__)

ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
ES_INDEX_NAME = os.getenv("ES_INDEX_NAME", "hybrid-docs")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "hybrid-docs")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

HYBRID_BM25_WEIGHT = os.getenv("HYBRID_BM25_WEIGHT", "0.4")
HYBRID_VECTOR_WEIGHT = os.getenv("HYBRID_VECTOR_WEIGHT", "")
HYBRID_TOP_K = os.getenv("HYBRID_TOP_K", "50")
HYBRID_RETURN_K = os.getenv("HYBRID_RETURN_K", "20")
HYBRID_FUSION_METHOD = os.getenv("HYBRID_FUSION_METHOD", "weighted_minmax")
HYBRID_RRF_K = os.getenv("HYBRID_RRF_K", "60")


_clients: Optional[Tuple[Elasticsearch, QdrantClient, OpenAI]] = None


def _get_clients() -> Tuple[Elasticsearch, QdrantClient, OpenAI]:
    global _clients
    if _clients is None:
        es = Elasticsearch(ELASTICSEARCH_URL)
        qdrant = QdrantClient(url=QDRANT_URL)
        openai_client = OpenAI()
        _clients = (es, qdrant, openai_client)
        log.info(
            "Initialized clients: es_index=%s qdrant_collection=%s embedding_model=%s",
            ES_INDEX_NAME,
            QDRANT_COLLECTION_NAME,
            OPENAI_EMBEDDING_MODEL,
        )
    return _clients

def normalize(scores: List[float]) -> List[float]:
    """Projects score distributions onto a [0, 1] range[cite: 38, 39]."""
    if not scores:
        return []
    mn, mx = min(scores), max(scores)
    if mx == mn:
        return [1.0] * len(scores)
    return [(s - mn) / (mx - mn) for s in scores]


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    return v


async def _embed_query(query: str) -> List[float]:
    _, _, openai_client = _get_clients()

    def _call() -> List[float]:
        r = openai_client.embeddings.create(model=OPENAI_EMBEDDING_MODEL, input=query)
        return r.data[0].embedding

    return await asyncio.to_thread(_call)


async def es_search(query: str, top_k: int) -> List[Dict]:
    """Standard BM25 keyword search."""
    es, _, _ = _get_clients()

    def _call():
        return es.search(
            index=ES_INDEX_NAME,
            query={"match": {"text": {"query": query}}},
            size=top_k,
        )

    response = await asyncio.to_thread(_call)
    hits = response.get("hits", {}).get("hits", [])
    out: List[Dict] = []
    for h in hits:
        src = h.get("_source", {}) or {}
        out.append(
            {
                "chunk_id": h.get("_id"),
                "score": float(h.get("_score") or 0.0),
                "text": src.get("text", ""),
                "source": src.get("source", ""),
            }
        )
    return out

async def vector_search(query: str, top_k: int) -> List[Dict]:
    """Semantic vector search (Qdrant)."""
    _, qdrant, _ = _get_clients()
    query_vector = await _embed_query(query)

    def _call():
        # qdrant-client v1.9+ uses query_points; older versions used search
        if hasattr(qdrant, "query_points"):
            return qdrant.query_points(
                collection_name=QDRANT_COLLECTION_NAME,
                query=query_vector,
                limit=top_k,
                with_payload=True,
            )
        return qdrant.search(  # type: ignore[attr-defined]
            collection_name=QDRANT_COLLECTION_NAME,
            query_vector=query_vector,
            limit=top_k,
        )

    response = await asyncio.to_thread(_call)
    # query_points returns an object with `.points`; search returns a list
    points = getattr(response, "points", response)
    out: List[Dict] = []
    for r in points:
        payload = getattr(r, "payload", None) or {}
        out.append(
            {
                "chunk_id": payload.get("chunk_id"),
                "score": float(getattr(r, "score", 0.0) or 0.0),
                "text": payload.get("text", ""),
                "source": payload.get("source", ""),
            }
        )
    return out

async def hybrid_search(
    query: str,
    bm25_weight: Optional[float] = None,
    vector_weight: Optional[float] = None,
    top_k: Optional[int] = None,
    return_k: Optional[int] = None,
    fusion_method: Optional[str] = None,
    rrf_k: Optional[int] = None,
    return_timings: bool = False,
) -> Union[List[Dict], Dict]:
    """
    Hybrid Retrieval Logic:
    1. Parallel Search
    2. Min-Max Normalization
    3. Fusion (weighted min-max or RRF baseline)
    """
    # Defaults from .env (callers can override by passing args)
    if top_k is None:
        top_k = _env_int("HYBRID_TOP_K", 50)
    if return_k is None:
        return_k = _env_int("HYBRID_RETURN_K", 20)

    if fusion_method is None:
        fusion_method = _env_str("HYBRID_FUSION_METHOD", "weighted_minmax").strip().lower()
    if rrf_k is None:
        rrf_k = _env_int("HYBRID_RRF_K", 60)

    if bm25_weight is None:
        bm25_weight = _env_float("HYBRID_BM25_WEIGHT", 0.4)
    if vector_weight is None:
        # If vector weight is not explicitly configured, use 1 - bm25_weight
        vector_weight = _env_float("HYBRID_VECTOR_WEIGHT", 1.0 - float(bm25_weight))

    # Keep weights sane; if misconfigured, fall back to (0.4, 0.6)
    if bm25_weight < 0.0 or vector_weight < 0.0 or (bm25_weight + vector_weight) <= 0.0:
        bm25_weight, vector_weight = 0.4, 0.6

    log.debug(
        "hybrid_search: fusion=%s top_k=%s return_k=%s bm25_w=%.3f vec_w=%.3f rrf_k=%s query_len=%d",
        fusion_method,
        top_k,
        return_k,
        float(bm25_weight),
        float(vector_weight),
        rrf_k,
        len(query),
    )

    # 1. Parallel search [cite: 45]
    async def _timed(coro):
        start = time.perf_counter()
        res = await coro
        return res, (time.perf_counter() - start) * 1000.0

    (bm25_results, es_ms), (vec_results, qdrant_ms) = await asyncio.gather(
        _timed(es_search(query, top_k)),
        _timed(vector_search(query, top_k)),
    )
    fusion_start = time.perf_counter()
    log.debug(
        "retrieval done: es=%d hits in %.1fms | qdrant=%d hits in %.1fms",
        len(bm25_results),
        es_ms,
        len(vec_results),
        qdrant_ms,
    )

    # 2. Normalize scores [cite: 46]
    bm25_scores = normalize([r["score"] for r in bm25_results])
    vec_scores = normalize([r["score"] for r in vec_results])

    # 3. Build lookups [cite: 47]
    bm25_map = {r["chunk_id"]: s for r, s in zip(bm25_results, bm25_scores)}
    vec_map = {r["chunk_id"]: s for r, s in zip(vec_results, vec_scores)}
    bm25_meta = {r["chunk_id"]: {"text": r.get("text", ""), "source": r.get("source", "")} for r in bm25_results}
    vec_meta = {r["chunk_id"]: {"text": r.get("text", ""), "source": r.get("source", "")} for r in vec_results}

    all_ids = set(bm25_map.keys()) | set(vec_map.keys())
    fused = []

    if fusion_method in {"rrf", "reciprocal_rank_fusion"}:
        bm25_rank = {r["chunk_id"]: i + 1 for i, r in enumerate(bm25_results)}
        vec_rank = {r["chunk_id"]: i + 1 for i, r in enumerate(vec_results)}
        for cid in all_ids:
            rrf_score = 0.0
            br = bm25_rank.get(cid)
            vr = vec_rank.get(cid)
            if br is not None:
                rrf_score += 1.0 / float(rrf_k + br)
            if vr is not None:
                rrf_score += 1.0 / float(rrf_k + vr)

            meta = bm25_meta.get(cid) or vec_meta.get(cid) or {}
            fused.append(
                {
                    "chunk_id": cid,
                    "hybrid_score": float(rrf_score),
                    "fusion_method": "rrf",
                    "rrf_k": int(rrf_k),
                    "bm25_rank": br,
                    "vec_rank": vr,
                    "bm25_score_norm": float(bm25_map.get(cid, 0.0) or 0.0),
                    "vec_score_norm": float(vec_map.get(cid, 0.0) or 0.0),
                    "text": meta.get("text", ""),
                    "source": meta.get("source", ""),
                }
            )
    else:
        # weighted min-max fusion (default)
        for cid in all_ids:
            bm25_norm = float(bm25_map.get(cid, 0.0) or 0.0)
            vec_norm = float(vec_map.get(cid, 0.0) or 0.0)
            score = (bm25_weight * bm25_norm) + (vector_weight * vec_norm)

            meta = bm25_meta.get(cid) or vec_meta.get(cid) or {}
            fused.append(
                {
                    "chunk_id": cid,
                    "hybrid_score": float(score),
                    "fusion_method": "weighted_minmax",
                    "bm25_weight": float(bm25_weight),
                    "vec_weight": float(vector_weight),
                    "bm25_score_norm": bm25_norm,
                    "vec_score_norm": vec_norm,
                    "text": meta.get("text", ""),
                    "source": meta.get("source", ""),
                }
            )

    # 5. Final Ranking [cite: 47]
    fused.sort(key=lambda x: x["hybrid_score"], reverse=True)
    fusion_ms = (time.perf_counter() - fusion_start) * 1000.0
    total_ms = es_ms + qdrant_ms + fusion_ms
    out = fused[:return_k]
    log.debug(
        "fusion done: method=%s candidates=%d out=%d fusion_ms=%.1f total_ms=%.1f",
        fusion_method,
        len(all_ids),
        len(out),
        fusion_ms,
        total_ms,
    )
    if return_timings:
        return {
            "results": out,
            "timings_ms": {
                "elasticsearch_ms": es_ms,
                "qdrant_ms": qdrant_ms,
                "fusion_ms": fusion_ms,
                "total_ms": total_ms,
            },
            "config": {
                "fusion_method": fusion_method,
                "bm25_weight": bm25_weight,
                "vector_weight": vector_weight,
                "top_k": top_k,
                "return_k": return_k,
                "rrf_k": rrf_k,
            },
        }
    return out  # Return top N for next stage (Re-ranking)