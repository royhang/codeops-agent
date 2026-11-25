from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from codeops_agent.agent.checkpoint import SQLiteCheckpointStore
from codeops_agent.agent.graph import MultiAgentGraph
from codeops_agent.agent.react import ReActEngine
from codeops_agent.context.manager import ContextManager
from codeops_agent.context.memory import DualLayerMemory
from codeops_agent.llm.base import BaseLLMClient, LLMResponse, Message, TokenUsage, ToolCall
from codeops_agent.llm.mock_client import RuleBasedMockLLMClient
from codeops_agent.observability.tracer import TraceRecorder
from codeops_agent.tools.builtin import build_default_registry

REPO_ROOT = Path(__file__).resolve().parents[1] / "benchmarks" / "sample_repo"
LOG_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "sample_logs" / "app.log"


class AlwaysToolCallLLM(BaseLLMClient):
    """Test double that endlessly requests tool calls to trigger circuit breaker."""

    async def chat(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.1,
        json_mode: bool = False,
    ) -> LLMResponse:
        _ = (messages, tools, temperature, json_mode)
        return LLMResponse(
            content="",
            tool_calls=[ToolCall(id="loop_1", name="inspect_logs", arguments={"keyword": "ERROR", "limit": 2})],
            usage=TokenUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30),
        )


@pytest.mark.asyncio
async def test_react_circuit_breaker_prevents_infinite_loop() -> None:
    registry, _ = build_default_registry(REPO_ROOT, LOG_PATH)
    engine = ReActEngine(llm=AlwaysToolCallLLM(), registry=registry, max_steps=3)
    outcome = await engine.run("Infinite loop test")
    assert outcome.hit_circuit_breaker is True
    assert outcome.steps_taken == 3
    assert len(outcome.records) == 3


@pytest.mark.asyncio
async def test_multi_agent_graph_end_to_end_and_checkpoint_resume(tmp_path: Path) -> None:
    db = tmp_path / "state.db"
    traces = tmp_path / "traces"
    registry, _ = build_default_registry(REPO_ROOT, LOG_PATH)

    graph = MultiAgentGraph(
        llm=RuleBasedMockLLMClient(),
        registry=registry,
        checkpoint_store=SQLiteCheckpointStore(db),
        memory=DualLayerMemory(db),
        context_manager=ContextManager(),
        tracer=TraceRecorder(traces),
        require_hitl_before_execute=True,
    )

    # First pass pauses at HITL approval gate after planning
    paused = await graph.run(
        query="多租户出现缓存串号，T_BETA 读到了 T_ALPHA 的订单摘要",
        task_id="task_hitl_01",
    )
    assert paused.status == "awaiting_approval"
    assert paused.awaiting_hitl is True
    assert len(paused.plan) >= 2

    # Resume from SQLite checkpoint with approval
    resumed = await graph.run(
        query="ignored on resume",
        task_id="task_hitl_01",
        resume_approval=True,
    )
    assert resumed.status == "completed"
    assert "app/cache.py" in resumed.final_answer
    assert resumed.review is not None and resumed.review.approved is True
    assert len(TraceRecorder(traces).list_recent_traces()) >= 1
