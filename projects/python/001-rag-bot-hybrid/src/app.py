import streamlit as st
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import os
from datetime import datetime, timezone
from uuid import uuid4
from statistics import mean
from typing import Any, Optional

import altair as alt

try:
    # If project root is on sys.path (package-style).
    from src.core_pipeline import (  # type: ignore
        generate_answer,
        judge_rag,
        judge_rag_ragas,
        search,
        _ragas_available,
    )
except ModuleNotFoundError:
    # When running `streamlit run src/app.py`, `src/` is the import root.
    from core_pipeline import (  # type: ignore
        generate_answer,
        judge_rag,
        judge_rag_ragas,
        search,
        _ragas_available,
    )

st.set_page_config(page_title="Hybrid RAG POC", layout="wide")

EVAL_LOG_PATH = os.getenv("EVAL_LOG_PATH", "./eval_runs.jsonl")

# Initialize Chat History in Streamlit Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for configuration
with st.sidebar:
    st.title("Settings")
    bm25_w = st.slider("BM25 Weight", 0.0, 1.0, 0.4)
    vec_w = 1.0 - bm25_w
    st.info(f"Vector Weight: {vec_w:.1f}")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

st.title("🚀 Hybrid Retrieval Chatbot")
st.caption("POC: Elasticsearch + Qdrant + LLM")


def append_jsonl(path: str, record: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    out: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def write_jsonl(path: str, records: list[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def safe_mean(vals: list[float]) -> float:
    return mean(vals) if vals else 0.0


def percentile(vals: list[float], p: float) -> float:
    """Nearest-rank percentile for small lists."""
    if not vals:
        return 0.0
    s = sorted(vals)
    k = int(round((p / 100.0) * (len(s) - 1)))
    k = max(0, min(len(s) - 1, k))
    return float(s[k])


def get_fusion_method(run: dict) -> str:
    cfg = run.get("config") or {}
    return str(cfg.get("fusion_method") or run.get("ab_tag") or "unknown")


def get_total_ms(run: dict) -> Optional[float]:
    t = (run.get("timings_ms") or {}).get("total_ms")
    return float(t) if isinstance(t, (int, float)) else None


def judged_runs(run_list: list[dict]) -> list[dict]:
    out = []
    for run in run_list:
        j = run.get("judge")
        if isinstance(j, dict) and j.get("error") is None:
            out.append(run)
    return out


def score_vals(run_list: list[dict], key: str) -> list[float]:
    vals: list[float] = []
    for run in run_list:
        j = run.get("judge") or {}
        v = j.get(key)
        if isinstance(v, (int, float)):
            vals.append(float(v))
    return vals


def halluc_rate(run_list: list[dict]) -> float:
    flags = []
    for run in run_list:
        j = run.get("judge") or {}
        h = j.get("hallucination_flag")
        if isinstance(h, bool):
            flags.append(h)
    return (sum(1 for x in flags if x) / len(flags)) if flags else 0.0


def run_coro(coro):
    """Run an async coroutine from Streamlit safely."""
    try:
        return asyncio.run(coro)
    except RuntimeError:
        # If an event loop is already running (rare in Streamlit), run in a thread.
        with ThreadPoolExecutor(max_workers=1) as ex:
            return ex.submit(lambda: asyncio.run(coro)).result()

# Tabs
chat_tab, eval_tab = st.tabs(["Chat", "Evaluation"])

with eval_tab:
    st.subheader("Evaluation Dashboard")
    runs = read_jsonl(EVAL_LOG_PATH)
    st.caption(f"Log file: `{EVAL_LOG_PATH}` • total runs: {len(runs)}")

    if not runs:
        st.info("No runs logged yet. Ask a question in the Chat tab to generate runs.")
    else:
        # Controls
        st.markdown("**Controls**")
        c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
        with c1:
            last_n = st.number_input(
                "Window (last N runs)",
                min_value=1,
                max_value=2000,
                value=min(50, max(1, len(runs))),
                step=10,
            )
        with c2:
            show_only_judged = st.checkbox("Show judged only", value=False)
        with c3:
            fusion_filter = st.selectbox(
                "Fusion filter",
                options=["(all)", "weighted_minmax", "rrf"],
                index=0,
            )
        with c4:
            rrf_k = st.number_input("RRF k (for A/B)", min_value=1, max_value=200, value=60, step=1)

        scoring_backend = st.radio(
            "Scoring backend",
            options=["custom", "ragas"],
            format_func=lambda x: "Ragas (faithfulness, answer_relevancy)" if x == "ragas" else "Custom (LLM judge)",
            index=1 if _ragas_available() else 0,
            horizontal=True,
        )
        if not _ragas_available() and scoring_backend == "ragas":
            st.caption("Ragas not installed. Install with: pip install ragas")

        window = runs[-int(last_n) :]
        if show_only_judged:
            window = judged_runs(window)
        if fusion_filter != "(all)":
            window = [r for r in window if get_fusion_method(r) == fusion_filter]

        judged = judged_runs(window)

        # Overview KPIs
        st.markdown("**Overview (selected window)**")
        latencies = [ms for r in window for ms in [get_total_ms(r)] if ms is not None]
        j_ground = score_vals(judged, "groundedness")
        j_cite = score_vals(judged, "citation_precision")
        j_help = score_vals(judged, "helpfulness")

        def kpi(label: str, value: str) -> None:
            st.markdown(f"<div style='font-size:0.9rem; color: #888'>{label}</div><div style='font-size:1.05rem'>{value}</div>", unsafe_allow_html=True)

        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            kpi("Runs", str(len(window)))
        with k2:
            kpi("Judged", f"{len(judged)} ({pct(len(judged)/max(1,len(window)))})")
        with k3:
            kpi("Groundedness", f"{safe_mean(j_ground):.2f}/5" if j_ground else "—")
        with k4:
            kpi("Citation precision", f"{safe_mean(j_cite):.2f}/5" if j_cite else "—")
        with k5:
            kpi("Hallucination rate", pct(halluc_rate(judged)) if judged else "—")

        # Latency block
        if latencies:
            l1, l2, l3 = st.columns(3)
            with l1:
                kpi("Latency p50", f"{percentile(latencies, 50):.0f} ms")
            with l2:
                kpi("Latency p95", f"{percentile(latencies, 95):.0f} ms")
            with l3:
                kpi("Latency avg", f"{safe_mean(latencies):.0f} ms")

        # Simple charts
        st.markdown("**Trends (by run order)**")
        ch1, ch2 = st.columns(2)
        with ch1:
            if judged:
                data = [
                    {"i": i, "metric": "groundedness", "value": v}
                    for i, v in enumerate(score_vals(judged, "groundedness"))
                ] + [
                    {"i": i, "metric": "citation_precision", "value": v}
                    for i, v in enumerate(score_vals(judged, "citation_precision"))
                ]
                chart = (
                    alt.Chart(alt.Data(values=data))
                    .mark_line()
                    .encode(x="i:Q", y="value:Q", color="metric:N")
                    .properties(height=220)
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.caption("No judged runs in this window.")
        with ch2:
            if latencies:
                data = [{"i": i, "total_ms": v} for i, v in enumerate(latencies)]
                chart = (
                    alt.Chart(alt.Data(values=data))
                    .mark_line()
                    .encode(x="i:Q", y="total_ms:Q")
                    .properties(height=220)
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.caption("No timing data in this window.")

        # Actions: judge + A/B compare
        st.markdown("**Actions**")
        a1, a2 = st.columns([1, 1])
        with a1:
            judge_btn = st.button("Judge (score) unjudged runs in window")
        with a2:
            ab_btn = st.button("A/B compare: weighted_minmax vs rrf (uses last N queries)")

        if judge_btn:
            updated = False
            use_ragas = scoring_backend == "ragas" and _ragas_available()
            for r in window:
                if r.get("judge"):
                    continue
                if use_ragas:
                    j = run_coro(
                        asyncio.to_thread(
                            judge_rag_ragas,
                            r.get("query", ""),
                            r.get("answer", ""),
                            (r.get("retrieval") or [])[:5],
                        )
                    )
                else:
                    j = run_coro(
                        judge_rag(
                            query=r.get("query", ""),
                            answer=r.get("answer", ""),
                            context_docs=(r.get("retrieval") or [])[:5],
                        )
                    )
                r["judge"] = j
                updated = True
            if updated:
                # persist full runs list (we mutated references)
                write_jsonl(EVAL_LOG_PATH, runs)
                st.success("Judging complete. Reloading.")
                st.rerun()
            st.info("Nothing to judge (all runs in window already have judge results).")

        if ab_btn:
            queries = [r.get("query", "") for r in runs[-int(last_n) :]]
            queries = [q for q in queries if q]
            if not queries:
                st.warning("No queries available to run A/B comparison.")
            else:
                st.warning("This will make OpenAI calls (answers + judge) and append new records to the log.")
                ab_records: list[dict] = []
                for q in queries:
                    for method in ["weighted_minmax", "rrf"]:
                        retrieval = run_coro(search(query=q, fusion_method=method, rrf_k=int(rrf_k), return_timings=True))
                        docs = retrieval["results"]
                        gen = run_coro(generate_answer(query=q, context_docs=docs[:5]))
                        ans = gen["answer"]
                        use_ragas_ab = scoring_backend == "ragas" and _ragas_available()
                        if use_ragas_ab:
                            judge = run_coro(asyncio.to_thread(judge_rag_ragas, q, ans, docs[:5]))
                        else:
                            judge = run_coro(judge_rag(query=q, answer=ans, context_docs=docs[:5]))
                        ab_records.append(
                            {
                                "run_id": str(uuid4()),
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "query": q,
                                "config": retrieval.get("config", {}),
                                "timings_ms": retrieval.get("timings_ms", {}),
                                "retrieval": docs,
                                "answer": ans,
                                "judge": judge,
                                "ab_tag": method,
                            }
                        )
                for rec in ab_records:
                    append_jsonl(EVAL_LOG_PATH, rec)
                st.success(f"Appended {len(ab_records)} A/B eval runs to log. Reloading.")
                st.rerun()

        # Runs table + drilldown
        st.markdown("**Runs (drill-down)**")
        table_rows: list[dict[str, Any]] = []
        for r in window[::-1][:200]:
            j = r.get("judge") or {}
            table_rows.append(
                {
                    "timestamp": r.get("timestamp"),
                    "fusion": get_fusion_method(r),
                    "query": (r.get("query") or "")[:120],
                    "grounded": j.get("groundedness"),
                    "cite": j.get("citation_precision"),
                    "help": j.get("helpfulness"),
                    "halluc": j.get("hallucination_flag"),
                    "total_ms": get_total_ms(r),
                    "run_id": r.get("run_id"),
                }
            )
        st.dataframe(table_rows, hide_index=True)

        run_ids = [r.get("run_id") for r in window if r.get("run_id")]
        selected_id = st.selectbox("Select a run_id to inspect", options=["(none)"] + run_ids, index=0)
        if selected_id != "(none)":
            rr = next((r for r in window if r.get("run_id") == selected_id), None)
            if rr:
                st.markdown("**Selected run details**")
                st.json(
                    {
                        "run_id": rr.get("run_id"),
                        "timestamp": rr.get("timestamp"),
                        "query": rr.get("query"),
                        "config": rr.get("config"),
                        "timings_ms": rr.get("timings_ms"),
                        "judge": rr.get("judge"),
                    }
                )
                with st.expander("Answer"):
                    st.write(rr.get("answer", ""))
                with st.expander("Top retrieved chunks (first 5)"):
                    for d in (rr.get("retrieval") or [])[:5]:
                        st.write(f"**{d.get('source','')}** • chunk_id={d.get('chunk_id')} • score={d.get('hybrid_score')}")
                        st.write((d.get("text") or "")[:800])
                        st.divider()

# Chat Input
with chat_tab:
    # 1) History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("View Retrieved Sources"):
                    for src in message["sources"]:
                        st.write(f"- **{src.get('source','')}**: {(src.get('text','') or '')[:200]}...")

    # 2) Current request/response (rendered after submit)
    if prompt := st.chat_input("Ask a question about your documents..."):
        # 1. Display User Message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Generate Assistant Response
        with st.chat_message("assistant"):
            with st.spinner("Searching and generating answer..."):
                # Step A: Hybrid Search (BM25 + Vector)
                retrieval = run_coro(
                    search(
                        query=prompt,
                        bm25_weight=bm25_w,
                        vector_weight=vec_w,
                        top_k=50,
                        return_k=20,
                        return_timings=True,
                    )
                )
                retrieved_docs = retrieval["results"]
                timings_ms = retrieval.get("timings_ms", {})
                retrieval_config = retrieval.get("config", {})

                # Step B: Grounded Generation (LLM)
                result = run_coro(generate_answer(query=prompt, context_docs=retrieved_docs[:5]))
                answer = result["answer"]
                sources = result["sources"]

                run_record = {
                    "run_id": str(uuid4()),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "query": prompt,
                    "config": {
                        **retrieval_config,
                        "bm25_weight": bm25_w,
                        "vector_weight": vec_w,
                    },
                    "timings_ms": timings_ms,
                    "retrieval": retrieved_docs,
                    "answer": answer,
                }
                append_jsonl(EVAL_LOG_PATH, run_record)

                # 3. Display Result & Sources
                st.markdown(answer)
                with st.expander("Sources Used"):
                    for doc in retrieved_docs[:5]:
                        st.write(f"**{doc.get('source','')}** (Score: {doc.get('hybrid_score',0.0):.3f})")
                        st.write(doc.get("text", ""))
                        st.divider()

        # 4. Save to History
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": retrieved_docs[:5],
            }
        )