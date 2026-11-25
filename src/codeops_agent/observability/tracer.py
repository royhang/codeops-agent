from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class SpanRecord(BaseModel):
    span_id: str = Field(default_factory=lambda: uuid4().hex[:12])
    name: str
    kind: str  # planner | executor | tool | reviewer | rag
    started_at: str
    duration_ms: float = 0.0
    status: str = "ok"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TraceRecord(BaseModel):
    trace_id: str = Field(default_factory=lambda: uuid4().hex[:16])
    task_id: str
    query: str
    started_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    total_duration_ms: float = 0.0
    total_tokens: int = 0
    status: str = "running"
    spans: List[SpanRecord] = Field(default_factory=list)


class TraceRecorder:
    """Records agent execution traces to local JSONL with optional LangFuse metadata export."""

    def __init__(self, trace_dir: Path | str = ".codeops_traces") -> None:
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.trace_file = self.trace_dir / "traces.jsonl"

    def start_trace(self, task_id: str, query: str) -> TraceRecord:
        return TraceRecord(task_id=task_id, query=query)

    def add_span(
        self,
        trace: TraceRecord,
        name: str,
        kind: str,
        duration_ms: float,
        status: str = "ok",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SpanRecord:
        span = SpanRecord(
            name=name,
            kind=kind,
            started_at=datetime.now(timezone.utc).isoformat(),
            duration_ms=round(duration_ms, 2),
            status=status,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            metadata=metadata or {},
        )
        trace.spans.append(span)
        trace.total_tokens += prompt_tokens + completion_tokens
        return span

    def finish_trace(self, trace: TraceRecord, start_perf: float, status: str = "completed") -> TraceRecord:
        trace.total_duration_ms = round((time.perf_counter() - start_perf) * 1000.0, 2)
        trace.status = status
        with self.trace_file.open("a", encoding="utf-8") as fh:
            fh.write(trace.model_dump_json() + "\n")
        return trace

    def list_recent_traces(self, limit: int = 20) -> List[Dict[str, Any]]:
        if not self.trace_file.is_file():
            return []
        lines = [ln for ln in self.trace_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
        recent = lines[-limit:]
        records: List[Dict[str, Any]] = []
        for ln in reversed(recent):
            try:
                records.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
        return records
