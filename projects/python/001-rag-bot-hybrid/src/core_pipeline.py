"""
`core_pipeline.py` is the thin orchestration layer used by the Streamlit UI.

It provides a stable interface for:
- **Retrieval**: call hybrid search (Elasticsearch BM25 + Qdrant vector) and return fused, ranked chunks.
- **Generation**: format a compact context from top chunks and ask an LLM to answer **only** from that context.

Why this file exists:
- Keeps `src/app.py` (UI) decoupled from retrieval/generation implementation details.
- Makes it easy to later swap fusion strategy, add re-ranking, change the LLM/provider, or adjust context formatting.

Key env vars:
- `OPENAI_CHAT_MODEL` (default: `gpt-4o-mini`): chat model used by `generate_answer`.

Contracts:
- `search(...)` returns a list of dicts with at least: `chunk_id`, `text`, `source`, `hybrid_score`.
- `generate_answer(...)` returns: `{"answer": str, "sources": context_docs}`.
"""

import os
import json
import logging
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from openai import OpenAI

try:
    # Package-style import.
    from .search import hybrid_search  # type: ignore
except ImportError:
    # Script-style import (e.g. when `src/` is on sys.path).
    from search import hybrid_search  # type: ignore

load_dotenv()

log = logging.getLogger(__name__)

OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_JUDGE_MODEL = os.getenv("OPENAI_JUDGE_MODEL", OPENAI_CHAT_MODEL)
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


async def search(
    query: str,
    bm25_weight: Optional[float] = None,
    vector_weight: Optional[float] = None,
    top_k: Optional[int] = None,
    return_k: Optional[int] = None,
    fusion_method: Optional[str] = None,
    rrf_k: Optional[int] = None,
    return_timings: bool = False,
) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
    return await hybrid_search(
        query=query,
        bm25_weight=bm25_weight,
        vector_weight=vector_weight,
        top_k=top_k,
        return_k=return_k,
        fusion_method=fusion_method,
        rrf_k=rrf_k,
        return_timings=return_timings,
    )


def _format_context(docs: List[Dict[str, Any]], max_chars: int = 12000) -> str:
    parts: List[str] = []
    total = 0
    for i, d in enumerate(docs, start=1):
        source = d.get("source", "")
        chunk_id = d.get("chunk_id", "")
        text = (d.get("text", "") or "").strip()
        if not text:
            continue
        block = f"[{i}] source={source} chunk_id={chunk_id}\n{text}"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts)


async def generate_answer(query: str, context_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Grounded answer generation: answer only using provided context.
    Returns: {"answer": str, "sources": [...]}
    """
    client = OpenAI()
    context = _format_context(context_docs)
    log.debug(
        "generate_answer: model=%s query_len=%d docs=%d context_chars=%d",
        OPENAI_CHAT_MODEL,
        len(query),
        len(context_docs),
        len(context),
    )

    system = (
        "You are a helpful assistant. Answer the user's question using ONLY the provided context. "
        "If the context does not contain the answer, say you don't have enough information."
    )
    user = f"Question:\n{query}\n\nContext:\n{context}"

    def _call() -> str:
        r = client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
        )
        return r.choices[0].message.content or ""

    # OpenAI SDK is sync; run in a thread so callers can stay async.
    import asyncio

    answer = await asyncio.to_thread(_call)
    return {"answer": answer.strip(), "sources": context_docs}


async def judge_rag(
    query: str,
    answer: str,
    context_docs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    LLM-as-judge for RAG quality. Returns structured metrics.

    Scores are 1-5 (higher is better) unless noted otherwise.
    """
    client = OpenAI()
    context = _format_context(context_docs, max_chars=12000)
    log.debug(
        "judge_rag: model=%s query_len=%d answer_len=%d docs=%d context_chars=%d",
        OPENAI_JUDGE_MODEL,
        len(query),
        len(answer),
        len(context_docs),
        len(context),
    )

    rubric = {
        "groundedness": "Does the answer follow ONLY from the provided context (no external facts)?",
        "citation_precision": "Are the retrieved chunks actually supporting the claims made?",
        "helpfulness": "Is the answer helpful and directly answers the question given the context?",
        "completeness": "Does it cover the key points that the context enables?",
        "refusal_correctness": "If context lacks the answer, does the assistant clearly say so?",
        "hallucination_flag": "Boolean: true if answer contains unsupported claims.",
    }

    system = (
        "You are a strict evaluator for a Retrieval-Augmented Generation system. "
        "Judge the assistant answer ONLY against the provided context. "
        "Return a single JSON object and nothing else."
    )
    user = (
        f"Question:\n{query}\n\n"
        f"Context:\n{context}\n\n"
        f"Assistant answer:\n{answer}\n\n"
        "Return JSON with keys:\n"
        "- groundedness (1-5)\n"
        "- citation_precision (1-5)\n"
        "- helpfulness (1-5)\n"
        "- completeness (1-5)\n"
        "- refusal_correctness (1-5)\n"
        "- hallucination_flag (true/false)\n"
        "- rationale (short string)\n"
        f"Rubric:\n{json.dumps(rubric, ensure_ascii=False)}"
    )

    def _call() -> str:
        r = client.chat.completions.create(
            model=OPENAI_JUDGE_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
        )
        return r.choices[0].message.content or ""

    import asyncio

    raw = (await asyncio.to_thread(_call)).strip()
    try:
        out = json.loads(raw)
        if isinstance(out, dict) and out.get("error") is None:
            log.debug(
                "judge_rag result: groundedness=%s citation_precision=%s hallucination_flag=%s",
                out.get("groundedness"),
                out.get("citation_precision"),
                out.get("hallucination_flag"),
            )
        return out
    except json.JSONDecodeError:
        # Best-effort fallback: wrap raw text
        log.warning("judge_rag: JSON parse failed; returning raw output (len=%d)", len(raw))
        return {"error": "judge_parse_failed", "raw": raw}


def _ragas_available() -> bool:
    try:
        import ragas  # noqa: F401
        return True
    except ImportError:
        return False


def judge_rag_ragas(
    query: str,
    answer: str,
    context_docs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Score RAG quality using Ragas (faithfulness, answer_relevancy).
    Returns a dict compatible with the Evaluation dashboard (same keys as judge_rag where possible).
    Ragas scores are 0-1; we also expose groundedness/helpfulness on 1-5 scale for UI consistency.
    """
    if not _ragas_available():
        return {"error": "ragas_unavailable", "message": "Install ragas: pip install ragas"}

    try:
        from ragas import EvaluationDataset, evaluate
        from ragas.llms import llm_factory
        from ragas.metrics import answer_relevancy, faithfulness
    except ImportError as e:
        log.warning("ragas import failed: %s", e)
        return {"error": "ragas_import_failed", "message": str(e)}

    contexts = [(d.get("text") or "").strip() for d in context_docs if (d.get("text") or "").strip()]
    if not contexts:
        return {"error": "ragas_no_context", "message": "No context text to evaluate"}

    # Provide explicit OpenAI-based LLM + embeddings so Ragas doesn't auto-pick other providers
    # and to satisfy metrics that require `embed_query` / `embed_documents`.
    openai_client = OpenAI()

    try:
        ragas_llm = llm_factory(OPENAI_JUDGE_MODEL, provider="openai", client=openai_client)
    except (ValueError, RuntimeError) as e:
        log.warning("ragas llm_factory failed: %s", e)
        return {"error": "ragas_llm_failed", "message": str(e)}

    class _OpenAIEmbeddingsAdapter:
        def __init__(self, client: OpenAI, model: str):
            self._client = client
            self._model = model

        def embed_query(self, text: str) -> List[float]:
            r = self._client.embeddings.create(model=self._model, input=text)
            return r.data[0].embedding

        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            if not texts:
                return []
            r = self._client.embeddings.create(model=self._model, input=texts)
            return [item.embedding for item in r.data]

    ragas_embeddings = _OpenAIEmbeddingsAdapter(openai_client, OPENAI_EMBEDDING_MODEL)

    sample = {
        "user_input": query,
        "retrieved_contexts": contexts,
        "response": answer,
    }
    try:
        dataset = EvaluationDataset.from_list([sample])
    except (ValueError, TypeError) as e:
        log.warning("ragas dataset build failed: %s", e)
        return {"error": "ragas_dataset_failed", "message": str(e)}

    try:
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            show_progress=False,
        )
    except (ValueError, RuntimeError) as e:
        log.warning("ragas evaluate failed: %s", e)
        return {"error": "ragas_evaluate_failed", "message": str(e)}

    # Normalize to our dashboard shape (0-1 from ragas; 1-5 for groundedness/helpfulness)
    out: Dict[str, Any] = {"provider": "ragas"}
    row: Dict[str, Any] = {}
    if hasattr(result, "scores") and result.scores is not None:
        try:
            sc = result.scores
            if hasattr(sc, "to_list"):
                scores_list = sc.to_list()
            elif hasattr(sc, "__getitem__"):
                scores_list = [sc[0]] if len(sc) > 0 else []
            else:
                scores_list = []
            row = scores_list[0] if scores_list and isinstance(scores_list[0], dict) else {}
        except (TypeError, ValueError, IndexError):
            pass
    if not row and hasattr(result, "__dict__"):
        row = {k: v for k, v in result.__dict__.items() if isinstance(k, str) and isinstance(v, (int, float))}

    f_val = row.get("faithfulness")
    ar_val = row.get("answer_relevancy")
    if isinstance(f_val, (int, float)):
        out["faithfulness"] = float(f_val)
        out["groundedness"] = round(1 + out["faithfulness"] * 4, 2)  # 1-5
        out["hallucination_flag"] = out["faithfulness"] < 0.5
    if isinstance(ar_val, (int, float)):
        out["answer_relevancy"] = float(ar_val)
        out["helpfulness"] = round(1 + out["answer_relevancy"] * 4, 2)  # 1-5
    out["citation_precision"] = out.get("groundedness")  # reuse for dashboard
    out["completeness"] = out.get("helpfulness")
    log.debug("judge_rag_ragas: faithfulness=%s answer_relevancy=%s", out.get("faithfulness"), out.get("answer_relevancy"))
    return out

