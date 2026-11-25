from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, List, Optional

from codeops_agent.agent.state import ExecutionRecord, StreamEvent
from codeops_agent.context.manager import ContextManager
from codeops_agent.llm.base import BaseLLMClient, Message, ToolCall
from codeops_agent.observability.tracer import TraceRecord, TraceRecorder
from codeops_agent.tools.registry import ToolRegistry

EventCallback = Callable[[StreamEvent], Awaitable[None]]


@dataclass
class ReActOutcome:
    answer: str
    records: List[ExecutionRecord]
    total_tokens: int
    steps_taken: int
    hit_circuit_breaker: bool = False


class ReActEngine:
    """Framework-free async ReAct execution loop with parallel tools, retries, and circuit breakers."""

    SYSTEM_PROMPT = (
        "ROLE: EXECUTOR\n"
        "You are CodeOps Executor, a senior SRE & software troubleshooting engineer.\n"
        "Always verify root causes using search_code, inspect_logs, and read_file before concluding.\n"
        "Cite exact file paths and symbol names observed in tool outputs."
    )

    def __init__(
        self,
        llm: BaseLLMClient,
        registry: ToolRegistry,
        context_manager: Optional[ContextManager] = None,
        tracer: Optional[TraceRecorder] = None,
        max_steps: int = 8,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.context_manager = context_manager or ContextManager()
        self.tracer = tracer
        self.max_steps = max_steps

    async def run(
        self,
        user_prompt: str,
        trace: Optional[TraceRecord] = None,
        on_event: Optional[EventCallback] = None,
    ) -> ReActOutcome:
        messages: List[Message] = [
            Message(role="system", content=self.SYSTEM_PROMPT),
            Message(role="user", content=user_prompt),
        ]
        tool_schemas = self.registry.openai_schemas()
        records: List[ExecutionRecord] = []
        total_tokens = 0

        for step in range(1, self.max_steps + 1):
            messages = await self.context_manager.compact_messages(messages, llm=self.llm)

            t0 = time.perf_counter()
            resp = await self.llm.chat(messages=messages, tools=tool_schemas)
            dur = (time.perf_counter() - t0) * 1000.0
            total_tokens += resp.usage.total_tokens

            if self.tracer and trace:
                self.tracer.add_span(
                    trace,
                    name=f"executor_step_{step}",
                    kind="executor",
                    duration_ms=dur,
                    prompt_tokens=resp.usage.prompt_tokens,
                    completion_tokens=resp.usage.completion_tokens,
                    metadata={"tool_calls": [tc.name for tc in resp.tool_calls]},
                )

            if not resp.tool_calls:
                return ReActOutcome(
                    answer=resp.content.strip(),
                    records=records,
                    total_tokens=total_tokens,
                    steps_taken=step,
                    hit_circuit_breaker=False,
                )

            messages.append(
                Message(role="assistant", content=resp.content, tool_calls=resp.tool_calls)
            )

            # Execute independent tool calls concurrently
            if on_event:
                for tc in resp.tool_calls:
                    await on_event(
                        StreamEvent(
                            event_type="tool_call",
                            node="executor",
                            data={"step": step, "tool": tc.name, "arguments": tc.arguments},
                        )
                    )

            async def _invoke(tc: ToolCall) -> tuple[ToolCall, str, bool, float]:
                t_tool = time.perf_counter()
                res = await self.registry.execute(tc.name, tc.arguments)
                t_dur = (time.perf_counter() - t_tool) * 1000.0
                model_text = self.context_manager.truncate_tool_output(res.to_model_text())
                return tc, model_text, res.ok, t_dur

            results = await asyncio.gather(*(_invoke(tc) for tc in resp.tool_calls))

            for tc, model_text, ok, t_dur in results:
                records.append(
                    ExecutionRecord(
                        step=step,
                        tool_name=tc.name,
                        arguments=tc.arguments,
                        ok=ok,
                        output_preview=model_text[:240],
                    )
                )
                messages.append(
                    Message(
                        role="tool",
                        name=tc.name,
                        tool_call_id=tc.id,
                        content=model_text,
                    )
                )
                if self.tracer and trace:
                    self.tracer.add_span(
                        trace,
                        name=f"tool:{tc.name}",
                        kind="tool",
                        duration_ms=t_dur,
                        status="ok" if ok else "error",
                        metadata={"arguments": json.dumps(tc.arguments, ensure_ascii=False)[:200]},
                    )
                if on_event:
                    await on_event(
                        StreamEvent(
                            event_type="tool_result",
                            node="executor",
                            data={
                                "step": step,
                                "tool": tc.name,
                                "ok": ok,
                                "preview": model_text[:220],
                            },
                        )
                    )

        # Circuit breaker fallback if max_steps reached without final text response
        return ReActOutcome(
            answer=(
                f"[Circuit Breaker] Reached max_steps={self.max_steps}. "
                f"Collected {len(records)} tool observations; please narrow the query or inspect traces."
            ),
            records=records,
            total_tokens=total_tokens,
            steps_taken=self.max_steps,
            hit_circuit_breaker=True,
        )
