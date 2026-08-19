"""Enterprise hybrid retrieval: multi-query, BM25, vector, RRF, Cohere/local rerank."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from pydantic import BaseModel, Field

from app import config

logger = logging.getLogger(__name__)

_local_reranker = None


class QueryVariations(BaseModel):
    queries: List[str] = Field(default_factory=list)


def _preview(text: str, limit: int = 160) -> str:
    clean = (text or "").replace("\n", " ").strip()
    if len(clean) > limit:
        return clean[: limit - 1].rstrip() + "…"
    return clean


def _chunk_id(doc: Document) -> str:
    return str(doc.metadata.get("chunk_id") or hash(doc.page_content))


def _hit_dict(doc: Document, rank: int) -> Dict[str, Any]:
    raw = ""
    try:
        import json

        original = json.loads(doc.metadata.get("original_content") or "{}")
        raw = original.get("raw_text") or ""
    except Exception:
        raw = ""
    return {
        "chunk_id": _chunk_id(doc),
        "rank": rank,
        "document_name": doc.metadata.get("document_name") or "",
        "preview": _preview(raw or doc.page_content),
    }


def generate_query_variations(question: str, llm) -> Tuple[List[str], int]:
    """Return [original, ...paraphrases] and elapsed ms."""
    t0 = time.perf_counter()
    variations: List[str] = []
    try:
        structured = llm.with_structured_output(QueryVariations)
        prompt = (
            f"Generate {config.MULTI_QUERY_N} different variations of this query "
            "that would help retrieve relevant document chunks.\n\n"
            f"Original query: {question}\n\n"
            "Return alternative queries that rephrase or approach the same "
            "question from different angles. Do not repeat the original."
        )
        response = structured.invoke(prompt)
        variations = [
            q.strip()
            for q in (response.queries or [])
            if q and q.strip() and q.strip().lower() != question.strip().lower()
        ][: config.MULTI_QUERY_N]
    except Exception as e:
        logger.warning("Multi-query generation failed, using original only: %s", e)

    queries = [question.strip()] + variations
    seen = set()
    unique: List[str] = []
    for q in queries:
        key = q.lower()
        if key not in seen:
            seen.add(key)
            unique.append(q)
    ms = int((time.perf_counter() - t0) * 1000)
    return unique, ms


def reciprocal_rank_fusion(
    ranked_lists: List[List[Document]],
    k: int = 60,
) -> Tuple[List[Tuple[Document, float]], List[Dict[str, Any]]]:
    """Fuse ranked lists with score = 1/(k + rank)."""
    scores: Dict[str, float] = defaultdict(float)
    docs_by_id: Dict[str, Document] = {}
    contributions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for list_idx, docs in enumerate(ranked_lists):
        for position, doc in enumerate(docs, start=1):
            cid = _chunk_id(doc)
            docs_by_id[cid] = doc
            add = 1.0 / (k + position)
            scores[cid] += add
            contributions[cid].append(
                {
                    "list_index": list_idx,
                    "rank": position,
                    "formula": f"1/({k}+{position})",
                    "add": round(add, 6),
                    "running_total": round(scores[cid], 6),
                }
            )

    sorted_pairs = sorted(
        ((docs_by_id[cid], score) for cid, score in scores.items()),
        key=lambda x: x[1],
        reverse=True,
    )
    contrib_log = [
        {
            "chunk_id": cid,
            "document_name": docs_by_id[cid].metadata.get("document_name") or "",
            "rrf_score": round(scores[cid], 6),
            "contributions": contributions[cid],
            "preview": _preview(
                docs_by_id[cid].metadata.get("bm25_text")
                or docs_by_id[cid].page_content
            ),
        }
        for cid, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    ]
    return sorted_pairs, contrib_log


def _get_local_reranker(model_name: str):
    global _local_reranker
    if (
        _local_reranker is not None
        and getattr(_local_reranker, "_model_name", None) == model_name
    ):
        return _local_reranker
    from sentence_transformers import CrossEncoder

    logger.info("Loading local reranker: %s", model_name)
    model = CrossEncoder(model_name)
    model._model_name = model_name  # type: ignore[attr-defined]
    _local_reranker = model
    return model


def _rerank_local(
    query: str, docs: List[Document], top_n: int, model_name: str
) -> Tuple[List[Document], List[Dict[str, Any]]]:
    if not docs:
        return [], []
    model = _get_local_reranker(model_name)
    pairs = [[query, d.page_content] for d in docs]
    scores = model.predict(pairs)
    scored = sorted(
        zip(docs, scores),
        key=lambda x: float(x[1]),
        reverse=True,
    )
    score_log = [
        {
            "chunk_id": _chunk_id(doc),
            "document_name": doc.metadata.get("document_name") or "",
            "score": round(float(score), 6),
            "preview": _preview(doc.page_content),
        }
        for doc, score in scored[:top_n]
    ]
    return [doc for doc, _ in scored[:top_n]], score_log


def _rerank_cohere(
    query: str, docs: List[Document], top_n: int, model_name: str, api_key: str
) -> Tuple[List[Document], List[Dict[str, Any]]]:
    import os

    from langchain_cohere import CohereRerank

    os.environ["COHERE_API_KEY"] = api_key
    compressor = CohereRerank(
        model=model_name,
        top_n=top_n,
        cohere_api_key=api_key,
    )
    reranked = list(compressor.compress_documents(docs, query))
    score_log = []
    for doc in reranked:
        relevance = doc.metadata.get("relevance_score")
        score_log.append(
            {
                "chunk_id": _chunk_id(doc),
                "document_name": doc.metadata.get("document_name") or "",
                "score": round(float(relevance), 6) if relevance is not None else None,
                "preview": _preview(doc.page_content),
            }
        )
    return reranked[:top_n], score_log


def rerank_documents(
    query: str, docs: List[Document], top_n: Optional[int] = None
) -> Tuple[List[Document], Dict[str, Any]]:
    """Rerank against the original user question. Cohere with local fallback."""
    top_n = top_n or config.RERANK_TOP_N
    meta: Dict[str, Any] = {
        "backend_requested": config.RERANK_BACKEND,
        "backend_used": None,
        "model": None,
        "fallback": False,
        "fallback_reason": None,
        "error": None,
        "input_size": len(docs),
        "scores": [],
    }
    if not docs:
        return [], meta

    use_local = config.RERANK_BACKEND == "local"
    if config.RERANK_BACKEND == "cohere" and not config.RERANK_API_KEY:
        use_local = True
        meta["fallback"] = True
        meta["fallback_reason"] = "no_api_key"

    if not use_local and config.RERANK_BACKEND == "cohere":
        try:
            ranked, scores = _rerank_cohere(
                query,
                docs,
                top_n,
                config.RERANK_MODEL,
                config.RERANK_API_KEY,
            )
            meta["backend_used"] = "cohere"
            meta["model"] = config.RERANK_MODEL
            meta["scores"] = scores
            return ranked, meta
        except Exception as e:
            logger.warning("Cohere rerank failed, falling back to local: %s", e)
            meta["error"] = str(e)[:300]
            meta["fallback"] = True
            meta["fallback_reason"] = str(e)[:200]
            use_local = True

    if config.RERANK_BACKEND == "local" and not meta["fallback"]:
        model_name = config.RERANK_MODEL or config.RERANK_FALLBACK_MODEL
    else:
        model_name = config.RERANK_FALLBACK_MODEL

    ranked, scores = _rerank_local(query, docs, top_n, model_name)
    meta["backend_used"] = "local"
    meta["model"] = model_name
    meta["scores"] = scores
    return ranked, meta


def retrieve_with_trace(
    question: str,
    *,
    vectorstore,
    bm25_retriever,
    llm,
) -> Tuple[List[Document], Dict[str, Any]]:
    """Hybrid retrieve with full trace for the Gradio log panel."""
    trace: Dict[str, Any] = {
        "original_query": question,
        "query_variations": [],
        "per_query": [],
        "rrf": [],
        "rerank": {},
        "timings_ms": {},
        "final_chunk_ids": [],
        "retrieved": 0,
    }

    queries, mq_ms = generate_query_variations(question, llm)
    trace["query_variations"] = queries
    trace["timings_ms"]["multi_query_ms"] = mq_ms

    vector_retriever = vectorstore.as_retriever(
        search_kwargs={"k": config.VECTOR_K}
    )

    ranked_lists: List[List[Document]] = []
    t_search = time.perf_counter()

    def _search_one(q: str) -> Tuple[str, List[Document], List[Document]]:
        bm25_docs: List[Document] = []
        vec_docs: List[Document] = []
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_bm25 = (
                pool.submit(bm25_retriever.invoke, q) if bm25_retriever else None
            )
            fut_vec = pool.submit(vector_retriever.invoke, q)
            if fut_bm25 is not None:
                try:
                    bm25_docs = list(fut_bm25.result())
                except Exception as e:
                    logger.warning("BM25 search failed for %r: %s", q, e)
            try:
                vec_docs = list(fut_vec.result())
            except Exception as e:
                logger.warning("Vector search failed for %r: %s", q, e)
        return q, bm25_docs, vec_docs

    with ThreadPoolExecutor(max_workers=min(4, max(1, len(queries)))) as pool:
        futures = [pool.submit(_search_one, q) for q in queries]
        results_by_query = {
            q: (bm, vec)
            for q, bm, vec in (f.result() for f in as_completed(futures))
        }

    for q in queries:
        bm25_docs, vec_docs = results_by_query.get(q, ([], []))
        ranked_lists.append(bm25_docs)
        ranked_lists.append(vec_docs)
        trace["per_query"].append(
            {
                "query": q,
                "bm25": [_hit_dict(d, i + 1) for i, d in enumerate(bm25_docs)],
                "vector": [_hit_dict(d, i + 1) for i, d in enumerate(vec_docs)],
            }
        )

    trace["timings_ms"]["search_ms"] = int((time.perf_counter() - t_search) * 1000)

    t_rrf = time.perf_counter()
    fused, contrib_log = reciprocal_rank_fusion(ranked_lists, k=config.RRF_K)
    top_fused = [doc for doc, _ in fused[: config.RRF_TOP_N]]
    trace["rrf"] = contrib_log[: config.RRF_TOP_N]
    trace["timings_ms"]["rrf_ms"] = int((time.perf_counter() - t_rrf) * 1000)

    if not top_fused:
        trace["retrieved"] = 0
        return [], trace

    t_rerank = time.perf_counter()
    final_docs, rerank_meta = rerank_documents(
        question, top_fused, config.RERANK_TOP_N
    )
    trace["rerank"] = rerank_meta
    trace["timings_ms"]["rerank_ms"] = int((time.perf_counter() - t_rerank) * 1000)

    trace["final_chunk_ids"] = [_chunk_id(d) for d in final_docs]
    trace["retrieved"] = len(final_docs)
    return final_docs, trace


def format_trace_markdown(trace: Dict[str, Any]) -> str:
    """Human-readable retrieval log for the Gradio panel."""
    lines: List[str] = []
    lines.append(f"## Query\n`{trace.get('original_query', '')}`\n")
    timings = trace.get("timings_ms") or {}
    lines.append(
        "**Timings (ms):** "
        f"multi_query={timings.get('multi_query_ms', '—')}, "
        f"search={timings.get('search_ms', '—')}, "
        f"rrf={timings.get('rrf_ms', '—')}, "
        f"rerank={timings.get('rerank_ms', '—')}\n"
    )

    variations = trace.get("query_variations") or []
    lines.append("### Multi-query variations")
    for i, q in enumerate(variations, 1):
        tag = " (original)" if i == 1 else ""
        lines.append(f"{i}. {q}{tag}")
    lines.append("")

    for block in trace.get("per_query") or []:
        lines.append(f"### Search for: `{block.get('query')}`")
        lines.append("**BM25**")
        for hit in block.get("bm25") or []:
            lines.append(
                f"- rank {hit['rank']} · `{hit.get('document_name')}` · "
                f"`{hit['chunk_id'][:12]}…` — {hit['preview']}"
            )
        if not block.get("bm25"):
            lines.append("- _(no hits)_")
        lines.append("**Vector**")
        for hit in block.get("vector") or []:
            lines.append(
                f"- rank {hit['rank']} · `{hit.get('document_name')}` · "
                f"`{hit['chunk_id'][:12]}…` — {hit['preview']}"
            )
        if not block.get("vector"):
            lines.append("- _(no hits)_")
        lines.append("")

    lines.append("### RRF fusion (top)")
    for i, row in enumerate(trace.get("rrf") or [], 1):
        lines.append(
            f"{i}. score **{row.get('rrf_score')}** · `{row.get('document_name')}` — "
            f"{row.get('preview')}"
        )
        for c in (row.get("contributions") or [])[:4]:
            lines.append(
                f"   - list#{c['list_index']} rank {c['rank']}: "
                f"{c['formula']} = {c['add']} → total {c['running_total']}"
            )
    if not trace.get("rrf"):
        lines.append("_No fused chunks._")
    lines.append("")

    rerank = trace.get("rerank") or {}
    lines.append("### Rerank")
    lines.append(
        f"- requested=`{rerank.get('backend_requested')}`, "
        f"used=`{rerank.get('backend_used')}`, model=`{rerank.get('model')}`"
    )
    lines.append(
        f"- fallback={rerank.get('fallback')}"
        + (
            f" ({rerank.get('fallback_reason')})"
            if rerank.get("fallback_reason")
            else ""
        )
    )
    if rerank.get("error"):
        lines.append(f"- error: `{rerank['error']}`")
    lines.append(
        f"- input_size={rerank.get('input_size')}, kept={trace.get('retrieved', 0)}"
    )
    for i, row in enumerate(rerank.get("scores") or [], 1):
        lines.append(
            f"{i}. score={row.get('score')} · `{row.get('document_name')}` — "
            f"{row.get('preview')}"
        )

    if trace.get("retrieved", 0) == 0:
        lines.append("\n**Result:** `retrieved=0` — no chunks sent to the LLM.")
    else:
        lines.append(
            f"\n**Final chunk ids:** {', '.join(trace.get('final_chunk_ids') or [])}"
        )

    lines.append("\n---\n")
    return "\n".join(lines)
