"""Query endpoint — full RAG pipeline."""

from __future__ import annotations

import dataclasses
import logging

from fastapi import APIRouter
from src.api.schemas import QueryRequest, QueryResponse
from src.confidence.fallback import route
from src.confidence.scorer import ConfidenceScorer
from src.confidence.threshold import ThresholdStrategy
from src.config import get_settings
from src.generation.context_builder import ContextBuilder
from src.generation.llm_client import LLMClient
from src.generation.prompt_manager import PromptManager
from src.index.embedding import EmbeddingEngine
from src.index.shared import bm25_index
from src.retrieval.query_rewriter import QueryRewriter
from src.retrieval.retriever import Retriever

logger = logging.getLogger(__name__)

router = APIRouter(tags=["query"])


# ---------------------------------------------------------------------------
# Module-level singletons for heavy dependencies
# Initialised once and reused across requests to avoid reloading models.
# Each factory function remains individually patchable in tests.
# ---------------------------------------------------------------------------
_retriever_instance: Retriever | None = None
_llm_client_instance: LLMClient | None = None
_confidence_scorer_instance: ConfidenceScorer | None = None
_threshold_instance: ThresholdStrategy | None = None
_prompt_manager_instance: PromptManager | None = None
_context_builder_instance: ContextBuilder | None = None
_query_rewriter_instance: QueryRewriter | None = None


def _build_retriever() -> Retriever:
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = Retriever(bm25_index=bm25_index)
    return _retriever_instance


def _build_embedding_engine() -> EmbeddingEngine:
    return EmbeddingEngine()


def _build_context_builder() -> ContextBuilder:
    global _context_builder_instance
    if _context_builder_instance is None:
        _context_builder_instance = ContextBuilder()
    return _context_builder_instance


def _build_llm_client() -> LLMClient:
    global _llm_client_instance
    if _llm_client_instance is None:
        _llm_client_instance = LLMClient()
    return _llm_client_instance


def _build_prompt_manager() -> PromptManager:
    global _prompt_manager_instance
    if _prompt_manager_instance is None:
        _prompt_manager_instance = PromptManager()
    return _prompt_manager_instance


def _build_confidence_scorer() -> ConfidenceScorer:
    global _confidence_scorer_instance
    if _confidence_scorer_instance is None:
        _confidence_scorer_instance = ConfidenceScorer()
    return _confidence_scorer_instance


def _build_threshold() -> ThresholdStrategy:
    global _threshold_instance
    if _threshold_instance is None:
        settings = get_settings()
        _threshold_instance = ThresholdStrategy(
            accept=settings.confidence_threshold_accept,
            reject=settings.confidence_threshold_reject,
        )
    return _threshold_instance


def _build_query_rewriter() -> QueryRewriter:
    global _query_rewriter_instance
    if _query_rewriter_instance is None:
        _query_rewriter_instance = QueryRewriter()
    return _query_rewriter_instance


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Run a RAG query against the indexed documents",
)
async def handle_query(request: QueryRequest) -> QueryResponse:
    """Execute the full RAG pipeline:

    1. Embed the query text
    2. Query rewriting for retrieval
    3. Hybrid search + reranker retrieval
    4. Context assembly with citations
    5. LLM answer generation
    6. Confidence scoring
    7. Threshold classification
    8. Fallback routing
    """
    # 1. Embedding
    embedding_engine = _build_embedding_engine()
    try:
        query_embedding = embedding_engine.embed(request.query)
    except Exception:
        logger.exception("Embedding generation failed, using zero vector")
        query_embedding = [0.0] * 1024
    finally:
        embedding_engine.unload()

    # 2. Query rewriting
    rewriter = _build_query_rewriter()
    rewritten_query = rewriter.rewrite(request.query)

    # 3. Retrieve (using rewritten query for search)
    retriever = _build_retriever()
    results = retriever.retrieve(
        rewritten_query,
        query_embedding,
        top_k=request.top_k,
    )

    # No results → early return
    if not results:
        return QueryResponse(
            answer="根据检索未找到相关文档内容，无法回答该问题。",
            citations=[],
            confidence=0.0,
            confidence_details={},
            needs_review=False,
        )

    # 4. Build context
    builder = _build_context_builder()
    context_str, citations = builder.build(results)

    # 5. Generate answer (using original query for the prompt)
    prompt_manager = _build_prompt_manager()
    prompt = prompt_manager.render("qa", query=request.query, context=context_str)

    llm = _build_llm_client()
    answer = llm.chat(prompt)

    if not answer:
        answer = "抱歉，AI 模型暂时无法生成回答。"

    # 6. Score confidence (query-time signals: reranker scores + result count)
    scorer = _build_confidence_scorer()
    reranker_scores = [r.score for r in results]
    score_result = scorer.score(
        reranker_scores=reranker_scores,
        num_results=len(results),
    )
    overall = score_result["overall"]
    details = score_result["details"]

    # 7. Threshold
    threshold = _build_threshold()
    decision = threshold.classify(overall)

    # 8. Route
    route_result = route(
        decision,
        {
            "answer": answer,
            "citations": [
                dataclasses.asdict(c) for c in citations
            ],
            "confidence": overall,
            "confidence_details": details,
        },
    )

    # 9. Build final response
    if route_result.decision == "reject":
        final_answer = "当前回答的置信度较低，建议重新提问或提供更详细的信息。"
        final_citations: list = []
    else:
        final_answer = answer
        final_citations = citations

    return QueryResponse(
        answer=final_answer,
        citations=final_citations,
        confidence=overall,
        confidence_details=details,
        needs_review=route_result.needs_review,
    )
