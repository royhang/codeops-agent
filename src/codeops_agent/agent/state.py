from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    step_id: int
    goal: str
    suggested_tool: str = "search_code"


class ExecutionRecord(BaseModel):
    step: int
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    ok: bool = True
    output_preview: str = ""


class ReviewFeedback(BaseModel):
    approved: bool
    confidence: float = 0.0
    critique: str = ""


class StreamEvent(BaseModel):
    """Structured SSE event emitted across Planner, Executor, Tool, and Reviewer stages."""

    event_type: str  # planner | tool_call | tool_result | reviewer | hitl_pause | final | error
    node: str
    data: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class AgentState(BaseModel):
    """Serializable StateGraph state persisted via SQLiteCheckpointStore."""

    task_id: str = Field(default_factory=lambda: uuid4().hex[:12])
    query: str
    status: str = "pending"  # pending | planned | executing | awaiting_approval | completed | failed
    plan: List[PlanStep] = Field(default_factory=list)
    execution_records: List[ExecutionRecord] = Field(default_factory=list)
    draft_answer: str = ""
    review: Optional[ReviewFeedback] = None
    final_answer: str = ""
    iteration: int = 0
    max_review_rounds: int = 2
    total_tokens: int = 0
    saved_tokens_est: int = 0
    trace_id: str = ""
    awaiting_hitl: bool = False
    hitl_approved: bool = True
