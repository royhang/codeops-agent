from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List

from pydantic import BaseModel

from codeops_agent.rag.chunker import chunk_repository
from codeops_agent.rag.hybrid import HybridRetriever, RetrievalHit


class RagStrategyMetrics(BaseModel):
    strategy: str
    total_queries: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr_at_5: float


class RagEvalSummary(BaseModel):
    total_queries: int
    strategies: List[RagStrategyMetrics]


def _load_golden(golden_path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for raw in golden_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def evaluate_rag(
    repo_root: Path,
    golden_path: Path,
) -> RagEvalSummary:
    """Benchmark BM25 vs Vector vs Hybrid(RRF) vs Hybrid+Rerank on rag_golden.jsonl."""
    chunks = chunk_repository(repo_root)
    retriever = HybridRetriever(chunks)
    cases = _load_golden(golden_path)

    strategies: Dict[str, Callable[[str, int], List[RetrievalHit]]] = {
        "BM25": lambda q, k: retriever.search_bm25(q, top_k=k),
        "Vector": lambda q, k: retriever.search_vector(q, top_k=k),
        "Hybrid(RRF)": lambda q, k: retriever.search_hybrid(q, top_k=k),
        "Hybrid+Rerank": lambda q, k: retriever.search_reranked(q, top_k=k),
    }

    results: List[RagStrategyMetrics] = []
    for name, search_fn in strategies.items():
        r1 = 0
        r3 = 0
        r5 = 0
        rr_sum = 0.0
        for case in cases:
            hits = search_fn(case["query"], 5)
            expected_file = case["expected_file"]
            expected_symbol = case["expected_symbol"]

            first_match_rank = None
            for idx, hit in enumerate(hits, start=1):
                if (
                    hit.chunk.file_path == expected_file
                    and hit.chunk.symbol_name == expected_symbol
                ):
                    first_match_rank = idx
                    break

            if first_match_rank is not None:
                if first_match_rank <= 1:
                    r1 += 1
                if first_match_rank <= 3:
                    r3 += 1
                if first_match_rank <= 5:
                    r5 += 1
                rr_sum += 1.0 / first_match_rank

        total = max(1, len(cases))
        results.append(
            RagStrategyMetrics(
                strategy=name,
                total_queries=len(cases),
                recall_at_1=round(r1 / total * 100.0, 1),
                recall_at_3=round(r3 / total * 100.0, 1),
                recall_at_5=round(r5 / total * 100.0, 1),
                mrr_at_5=round(rr_sum / total, 4),
            )
        )

    return RagEvalSummary(total_queries=len(cases), strategies=results)
