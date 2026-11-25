from __future__ import annotations

import json
import time
from typing import AsyncGenerator, Optional

from codeops_agent.agent.checkpoint import SQLiteCheckpointStore
from codeops_agent.agent.react import EventCallback, ReActEngine
from codeops_agent.agent.state import (
    AgentState,
    PlanStep,
    ReviewFeedback,
    StreamEvent,
)
from codeops_agent.context.manager import ContextManager
from codeops_agent.context.memory import DualLayerMemory
from codeops_agent.llm.base import BaseLLMClient, Message
from codeops_agent.observability.tracer import TraceRecorder
from codeops_agent.tools.registry import ToolRegistry


class MultiAgentGraph:
    """StateGraph orchestrator implementing Planner -> Executor -> Reviewer with checkpoints & HITL."""

    PLANNER_PROMPT = (
        "ROLE: PLANNER\n"
        "Decompose the user's code/log troubleshooting request into 2-4 concise steps.\n"
        'Return JSON: {"plan": [{"step_id": 1, "goal": "...", "suggested_tool": "..."}]}.'
    )

    REVIEWER_PROMPT = (
        "ROLE: REVIEWER\n"
        "Verify whether the draft diagnosis cites specific file paths/symbols and actionable fixes.\n"
        'Return JSON: {"approved": true/false, "confidence": 0.0-1.0, "critique": "..."}.'
    )

    def __init__(
        self,
        llm: BaseLLMClient,
        registry: ToolRegistry,
        checkpoint_store: Optional[SQLiteCheckpointStore] = None,
        memory: Optional[DualLayerMemory] = None,
        context_manager: Optional[ContextManager] = None,
        tracer: Optional[TraceRecorder] = None,
        max_steps: int = 8,
        require_hitl_before_execute: bool = False,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.checkpoint_store = checkpoint_store or SQLiteCheckpointStore()
        self.memory = memory or DualLayerMemory()
        self.context_manager = context_manager or ContextManager()
        self.tracer = tracer or TraceRecorder()
        self.max_steps = max_steps
        self.require_hitl_before_execute = require_hitl_before_execute
        self.react_engine = ReActEngine(
            llm=llm,
            registry=registry,
            context_manager=self.context_manager,
            tracer=self.tracer,
            max_steps=max_steps,
        )

    async def run(
        self,
        query: str,
        task_id: Optional[str] = None,
        resume_approval: Optional[bool] = None,
        on_event: Optional[EventCallback] = None,
    ) -> AgentState:
        start_perf = time.perf_counter()

        existing = self.checkpoint_store.load(task_id) if task_id else None
        if existing is not None:
            state = existing
            if resume_approval is not None:
                state.hitl_approved = resume_approval
                state.awaiting_hitl = False
        else:
            state = AgentState(query=query)
            if task_id:
                state.task_id = task_id

        trace = self.tracer.start_trace(task_id=state.task_id, query=state.query)
        state.trace_id = trace.trace_id

        async def _emit(ev: StreamEvent) -> None:
            if on_event is not None:
                await on_event(ev)

        # Node 1: Planner (skip if already planned during checkpoint resume)
        if not state.plan:
            state.plan = await self._run_planner(state, trace)
            state.status = "planned"
            self.checkpoint_store.save(state)
            await _emit(
                StreamEvent(
                    event_type="planner",
                    node="planner",
                    data={"steps": [s.model_dump() for s in state.plan]},
                )
            )

        # Optional Human-in-the-Loop Gate before Execution
        if self.require_hitl_before_execute and resume_approval is None and state.status == "planned":
            state.status = "awaiting_approval"
            state.awaiting_hitl = True
            self.checkpoint_store.save(state)
            await _emit(
                StreamEvent(
                    event_type="hitl_pause",
                    node="hitl_gate",
                    data={"task_id": state.task_id, "message": "Plan ready; awaiting human approval."},
                )
            )
            self.tracer.finish_trace(trace, start_perf, status="paused")
            return state

        if not state.hitl_approved:
            state.status = "failed"
            state.final_answer = "Task aborted by human reviewer at approval gate."
            self.checkpoint_store.save(state)
            self.tracer.finish_trace(trace, start_perf, status="aborted")
            return state

        # Node 2 & 3 Loop: Executor -> Reviewer
        recalled_facts = self.memory.recall_long_term("troubleshooting", state.query, limit=2)
        memory_hint = ""
        if recalled_facts:
            memory_hint = "\nRelevant historical findings:\n" + "\n".join(
                f"- {f.fact_key}: {f.content}" for f in recalled_facts
            )

        critique_hint = ""
        while state.iteration < state.max_review_rounds:
            state.iteration += 1
            state.status = "executing"

            plan_text = "\n".join(f"{s.step_id}. {s.goal} (tool: {s.suggested_tool})" for s in state.plan)
            exec_prompt = (
                f"Task: {state.query}\n\n"
                f"Execution Plan:\n{plan_text}"
                f"{memory_hint}"
                f"{critique_hint}"
            )
            outcome = await self.react_engine.run(exec_prompt, trace=trace, on_event=on_event)
            state.execution_records.extend(outcome.records)
            state.draft_answer = outcome.answer
            state.total_tokens += outcome.total_tokens
            state.saved_tokens_est = self.context_manager.stats.saved_tokens_est
            self.checkpoint_store.save(state)

            # Node 3: Reviewer
            review = await self._run_reviewer(state, trace)
            state.review = review
            await _emit(
                StreamEvent(
                    event_type="reviewer",
                    node="reviewer",
                    data=review.model_dump(),
                )
            )

            if review.approved or outcome.hit_circuit_breaker:
                break
            critique_hint = f"\n\nReviewer Critique (fix in next pass): {review.critique}"

        state.final_answer = state.draft_answer
        state.status = "completed"
        self.checkpoint_store.save(state)

        # Persist distilled finding into DualLayerMemory
        self.memory.add_short_term(f"Q: {state.query[:80]} -> {state.final_answer[:120]}")
        self.memory.upsert_long_term(
            namespace="troubleshooting",
            fact_key=state.task_id,
            content=state.final_answer[:280],
        )

        self.tracer.finish_trace(trace, start_perf, status="completed")
        await _emit(
            StreamEvent(
                event_type="final",
                node="graph",
                data={
                    "task_id": state.task_id,
                    "status": state.status,
                    "final_answer": state.final_answer,
                    "total_tokens": state.total_tokens,
                    "saved_tokens_est": state.saved_tokens_est,
                    "trace_id": state.trace_id,
                },
            )
        )
        return state

    async def stream(self, query: str, task_id: Optional[str] = None) -> AsyncGenerator[StreamEvent, None]:
        """Async generator yielding live StreamEvents for FastAPI SSE endpoints."""
        import asyncio

        queue: asyncio.Queue[Optional[StreamEvent]] = asyncio.Queue()

        async def _push(ev: StreamEvent) -> None:
            await queue.put(ev)

        async def _runner() -> None:
            try:
                await self.run(query=query, task_id=task_id, on_event=_push)
            except Exception as exc:
                await queue.put(
                    StreamEvent(event_type="error", node="graph", data={"error": str(exc)})
                )
            finally:
                await queue.put(None)

        task = asyncio.create_task(_runner())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item
        finally:
            await task

    async def _run_planner(self, state: AgentState, trace: object) -> list[PlanStep]:
        t0 = time.perf_counter()
        resp = await self.llm.chat(
            messages=[
                Message(role="system", content=self.PLANNER_PROMPT),
                Message(role="user", content=f"Task: {state.query}"),
            ],
            json_mode=True,
        )
        dur = (time.perf_counter() - t0) * 1000.0
        state.total_tokens += resp.usage.total_tokens
        self.tracer.add_span(
            trace,  # type: ignore[arg-type]
            name="planner_decompose",
            kind="planner",
            duration_ms=dur,
            prompt_tokens=resp.usage.prompt_tokens,
            completion_tokens=resp.usage.completion_tokens,
        )
        try:
            data = json.loads(resp.content)
            return [PlanStep.model_validate(item) for item in data.get("plan", [])]
        except Exception:
            return [
                PlanStep(step_id=1, goal=f"Search logs and codebase for: {state.query[:60]}", suggested_tool="search_code"),
                PlanStep(step_id=2, goal="Read top matching file and report root cause", suggested_tool="read_file"),
            ]

    async def _run_reviewer(self, state: AgentState, trace: object) -> ReviewFeedback:
        t0 = time.perf_counter()
        resp = await self.llm.chat(
            messages=[
                Message(role="system", content=self.REVIEWER_PROMPT),
                Message(
                    role="user",
                    content=f"Task: {state.query}\n\nDraft Answer:\n{state.draft_answer}",
                ),
            ],
            json_mode=True,
        )
        dur = (time.perf_counter() - t0) * 1000.0
        state.total_tokens += resp.usage.total_tokens
        self.tracer.add_span(
            trace,  # type: ignore[arg-type]
            name="reviewer_verify",
            kind="reviewer",
            duration_ms=dur,
            prompt_tokens=resp.usage.prompt_tokens,
            completion_tokens=resp.usage.completion_tokens,
        )
        try:
            return ReviewFeedback.model_validate_json(resp.content)
        except Exception:
            return ReviewFeedback(approved=True, confidence=0.75, critique="Fallback approval.")
