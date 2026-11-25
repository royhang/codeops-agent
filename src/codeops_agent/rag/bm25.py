from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Tuple

from codeops_agent.rag.chunker import CodeChunk


def code_tokenize(text: str) -> List[str]:
    """Code-aware tokenizer splitting snake_case, camelCase, paths, and symbols."""
    # Split camelCase before lowercasing
    expanded = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    raw_tokens = re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+", expanded)
    tokens: List[str] = []
    for tok in raw_tokens:
        low = tok.lower()
        tokens.append(low)
        if "_" in low:
            parts = [p for p in low.split("_") if len(p) > 1]
            tokens.extend(parts)
    return tokens


class BM25Retriever:
    """Pure-Python BM25Okapi retriever optimized for code & log search."""

    def __init__(self, chunks: List[CodeChunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self._doc_tokens: List[List[str]] = []
        self._doc_freqs: List[Counter] = []
        self._idf: Dict[str, float] = {}
        self._avgdl: float = 1.0
        self._build_index()

    def _build_index(self) -> None:
        n_docs = len(self.chunks)
        if n_docs == 0:
            return
        df: Counter = Counter()
        total_len = 0
        for chunk in self.chunks:
            # Give symbol & file_path slightly higher presence by including them explicitly
            doc_text = f"{chunk.file_path} {chunk.symbol_name} {chunk.content}"
            toks = code_tokenize(doc_text)
            self._doc_tokens.append(toks)
            freqs = Counter(toks)
            self._doc_freqs.append(freqs)
            total_len += len(toks)
            for term in freqs.keys():
                df[term] += 1

        self._avgdl = max(1.0, total_len / n_docs)
        for term, freq in df.items():
            self._idf[term] = math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))

    def search(self, query: str, top_k: int = 5) -> List[Tuple[CodeChunk, float]]:
        if not self.chunks:
            return []
        q_tokens = code_tokenize(query)
        scores: List[Tuple[int, float]] = []
        for idx, freqs in enumerate(self._doc_freqs):
            doc_len = len(self._doc_tokens[idx])
            score = 0.0
            for term in q_tokens:
                if term not in freqs:
                    continue
                tf = freqs[term]
                idf = self._idf.get(term, 0.0)
                denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self._avgdl))
                score += idf * ((tf * (self.k1 + 1.0)) / max(1e-9, denom))
            if score > 0.0:
                scores.append((idx, score))

        scores.sort(key=lambda item: item[1], reverse=True)
        return [(self.chunks[idx], round(score, 6)) for idx, score in scores[:top_k]]
