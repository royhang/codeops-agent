from __future__ import annotations

from pathlib import Path

import pytest

from codeops_agent.context.manager import ContextManager
from codeops_agent.context.memory import DualLayerMemory
from codeops_agent.llm.base import Message
from codeops_agent.llm.mock_client import RuleBasedMockLLMClient


def test_context_manager_truncates_verbose_tool_output() -> None:
    cm = ContextManager(max_tool_output_chars=200)
    raw = "HEADER_LINE\n" + ("noise " * 200) + "\nTAIL_ERROR_LINE"
    out = cm.truncate_tool_output(raw)

    assert len(out) < len(raw)
    assert "HEADER_LINE" in out
    assert "TAIL_ERROR_LINE" in out
    assert cm.stats.saved_chars > 0
    assert cm.stats.saved_tokens_est > 0


@pytest.mark.asyncio
async def test_context_manager_rolling_summary_compaction() -> None:
    cm = ContextManager(token_budget=100, keep_recent_messages=2)
    llm = RuleBasedMockLLMClient()
    messages = [
        Message(role="system", content="System instruction."),
        Message(role="user", content="Initial bug report: investigate pricing error."),
        Message(role="assistant", content="Step 1 checking logs " * 20),
        Message(role="tool", content="Tool log output details " * 20),
        Message(role="assistant", content="Step 2 reading pricing.py " * 20),
        Message(role="tool", content="Recent tool output"),
        Message(role="assistant", content="Latest reply"),
    ]
    compacted = await cm.compact_messages(messages, llm=llm)
    assert len(compacted) < len(messages)
    assert any("Rolling Context Summary" in m.content for m in compacted)
    assert cm.stats.compression_events >= 1


def test_dual_layer_memory_persistence(tmp_path: Path) -> None:
    db_file = tmp_path / "memory.db"
    mem = DualLayerMemory(db_path=db_file, short_term_limit=2)
    mem.add_short_term("turn 1")
    mem.add_short_term("turn 2")
    mem.add_short_term("turn 3")
    assert mem.get_short_term() == ["turn 2", "turn 3"]

    mem.upsert_long_term("troubleshooting", "pricing_bug", "app/pricing.py ZeroDivisionError when gross_total == 0")
    recalled = mem.recall_long_term("troubleshooting", "ZeroDivisionError pricing.py", limit=2)
    assert recalled
    assert recalled[0].fact_key == "pricing_bug"
