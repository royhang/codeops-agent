from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel

from codeops_agent.rag.bm25 import BM25Retriever, code_tokenize
from codeops_agent.rag.chunker import CodeChunk
from codeops_agent.rag.vector import BaseEmbedder, VectorRetriever


class RetrievalHit(BaseModel):
    chunk: CodeChunk
    score: float
    source: str  # bm25 | vector | hybrid_rrf | hybrid_rerank


class HybridRetriever:
    """Combines BM25 + Vector retrieval via Reciprocal Rank Fusion (RRF) and Reranking."""

    def __init__(
        self,
        chunks: List[CodeChunk],
        embedder: Optional[BaseEmbedder] = None,
        rrf_k: int = 60,
    ) -> None:
        self.chunks = chunks
        self.bm25 = BM25Retriever(chunks)
        self.vector = VectorRetriever(chunks, embedder=embedder)
        self.rrf_k = rrf_k
        self._chunk_map: Dict[str, CodeChunk] = {c.chunk_id: c for c in chunks}

    def search_bm25(self, query: str, top_k: int = 5) -> List[RetrievalHit]:
        return [
            RetrievalHit(chunk=c, score=s, source="bm25")
            for c, s in self.bm25.search(query, top_k=top_k)
        ]

    def search_vector(self, query: str, top_k: int = 5) -> List[RetrievalHit]:
        return [
            RetrievalHit(chunk=c, score=s, source="vector")
            for c, s in self.vector.search(query, top_k=top_k)
        ]

    def search_hybrid(self, query: str, top_k: int = 5, candidate_k: int = 12) -> List[RetrievalHit]:
        bm25_hits = self.bm25.search(query, top_k=candidate_k)
        vec_hits = self.vector.search(query, top_k=candidate_k)

        rrf_scores: Dict[str, float] = {}
        for rank, (chunk, _) in enumerate(bm25_hits, start=1):
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + 1.0 / (self.rrf_k + rank)
        for rank, (chunk, _) in enumerate(vec_hits, start=1):
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + 1.0 / (self.rrf_k + rank)

        ordered = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [
            RetrievalHit(chunk=self._chunk_map[cid], score=round(sc, 6), source="hybrid_rrf")
            for cid, sc in ordered[:top_k]
        ]

    def search_reranked(self, query: str, top_k: int = 5, candidate_k: int = 12) -> List[RetrievalHit]:
        """Rerank RRF candidates using symbol-level alignment, token overlap, and AST granularity."""
        candidates = self.search_hybrid(query, top_k=candidate_k, candidate_k=candidate_k)
        q_lower = query.lower()
        q_tokens = set(code_tokenize(query))

        rescored: List[Tuple[CodeChunk, float]] = []
        for hit in candidates:
            c = hit.chunk
            score = hit.score * 30.0

            # Exact symbol name match is the strongest signal in code QA
            sym_low = c.symbol_name.lower()
            if sym_low and sym_low in q_lower:
                score += 1.6
            sym_tokens = set(code_tokenize(c.symbol_name))
            if q_tokens and sym_tokens:
                sym_overlap = len(q_tokens & sym_tokens) / max(1, len(sym_tokens))
                score += sym_overlap * 1.1

            file_stem = c.file_path.rsplit("/", 1)[-1].replace(".py", "").lower()
            if file_stem and file_stem in q_lower:
                score += 0.45

            # Token overlap density in chunk body
            doc_tokens = set(code_tokenize(c.content))
            if q_tokens:
                overlap_ratio = len(q_tokens & doc_tokens) / len(q_tokens)
                score += overlap_ratio * 0.5

            # Prefer granular function/method chunks over broad enclosing class chunks
            if c.kind == "function":
                score += 0.35
            elif c.kind == "class":
                score += 0.10

            rescored.append((c, round(score, 6)))

        rescored.sort(key=lambda item: item[1], reverse=True)
        return [
            RetrievalHit(chunk=c, score=sc, source="hybrid_rerank")
            for c, sc in rescored[:top_k]
        ]
