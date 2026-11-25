from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from typing import List, Tuple

from codeops_agent.rag.bm25 import code_tokenize
from codeops_agent.rag.chunker import CodeChunk


class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self, text: str) -> List[float]:
        raise NotImplementedError


class LocalHashEmbedder(BaseEmbedder):
    """Deterministic local dense embedder combining token & char-ngram hashing (L2-normalized).

    Enables semantic-like fuzzy vector similarity offline with zero external model download.
    Can be swapped for OpenAI/BGE embedder implementing BaseEmbedder.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _hash_bucket(self, feature: str) -> Tuple[int, float]:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        val = int.from_bytes(digest, "big")
        bucket = val % self.dim
        sign = 1.0 if ((val >> 8) & 1) == 0 else -1.0
        return bucket, sign

    def embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        tokens = code_tokenize(text)
        for tok in tokens:
            b, s = self._hash_bucket(f"tok:{tok}")
            vec[b] += s * 1.5
            # Character 3-grams catch typos / partial identifiers (e.g., ORDER_DB_URl vs ORDER_DB_URL)
            if len(tok) >= 3:
                for i in range(len(tok) - 2):
                    ng = tok[i : i + 3]
                    nb, ns = self._hash_bucket(f"ng:{ng}")
                    vec[nb] += ns * 0.45

        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0.0:
            vec = [v / norm for v in vec]
        return vec


def cosine_similarity(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class VectorRetriever:
    """In-memory dense vector retriever with pluggable BaseEmbedder."""

    def __init__(self, chunks: List[CodeChunk], embedder: BaseEmbedder | None = None) -> None:
        self.chunks = chunks
        self.embedder = embedder or LocalHashEmbedder()
        self._vectors: List[List[float]] = [
            self.embedder.embed(f"{c.file_path} {c.symbol_name} {c.content}") for c in chunks
        ]

    def search(self, query: str, top_k: int = 5) -> List[Tuple[CodeChunk, float]]:
        if not self.chunks:
            return []
        q_vec = self.embedder.embed(query)
        scored = [
            (idx, cosine_similarity(q_vec, doc_vec))
            for idx, doc_vec in enumerate(self._vectors)
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return [(self.chunks[idx], round(score, 6)) for idx, score in scored[:top_k]]
