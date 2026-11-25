from __future__ import annotations

from pathlib import Path

from codeops_agent.eval.rag_eval import evaluate_rag
from codeops_agent.rag.chunker import chunk_repository
from codeops_agent.rag.hybrid import HybridRetriever

REPO_ROOT = Path(__file__).resolve().parents[1] / "benchmarks" / "sample_repo"
RAG_GOLDEN = Path(__file__).resolve().parents[1] / "benchmarks" / "rag_golden.jsonl"


def test_ast_chunking_extracts_functions_and_classes() -> None:
    chunks = chunk_repository(REPO_ROOT)
    symbols = {c.symbol_name for c in chunks}
    assert "calculate_discount_rate" in symbols
    assert "TenantCache" in symbols
    assert "build_cache_key" in symbols
    assert "is_token_valid" in symbols
    assert "OrderRepository" in symbols
    assert "PaymentGatewayClient" in symbols
    assert "load_service_config" in symbols


def test_hybrid_rerank_retrieves_expected_symbols() -> None:
    chunks = chunk_repository(REPO_ROOT)
    retriever = HybridRetriever(chunks)

    hits = retriever.search_reranked("TenantCache build_cache_key tenant_id collision", top_k=3)
    assert hits
    assert hits[0].chunk.file_path == "app/cache.py"
    assert hits[0].chunk.symbol_name == "build_cache_key"


def test_rag_golden_benchmark_high_recall() -> None:
    summary = evaluate_rag(REPO_ROOT, RAG_GOLDEN)
    by_name = {s.strategy: s for s in summary.strategies}
    assert summary.total_queries == 12
    assert by_name["Hybrid+Rerank"].recall_at_5 >= 90.0
    assert by_name["Hybrid+Rerank"].mrr_at_5 >= by_name["Vector"].mrr_at_5
