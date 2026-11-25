from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel

from codeops_agent.agent.checkpoint import SQLiteCheckpointStore
from codeops_agent.agent.graph import MultiAgentGraph
from codeops_agent.context.manager import ContextManager
from codeops_agent.context.memory import DualLayerMemory
from codeops_agent.eval.metrics import percentile
from codeops_agent.llm.base import BaseLLMClient
from codeops_agent.llm.mock_client import RuleBasedMockLLMClient
from codeops_agent.observability.tracer import TraceRecorder
from codeops_agent.tools.builtin import build_default_registry


class AgentCaseResult(BaseModel):
    case_id: str
    passed: bool
    file_grounded: bool
    keywords_matched: int
    keywords_total: int
    required_tools_used: bool
    latency_ms: float
    total_tokens: int
    bad_case_reason: str = ""


class AgentEvalReport(BaseModel):
    total_cases: int
    passed_cases: int
    pass_rate: float
    tool_grounding_rate: float
    avg_tokens: float
    p50_latency_ms: float
    p95_latency_ms: float
    bad_cases: List[AgentCaseResult]
    results: List[AgentCaseResult]


async def evaluate_agent(
    repo_root: Path,
    log_path: Optional[Path],
    golden_path: Path,
    llm: Optional[BaseLLMClient] = None,
) -> AgentEvalReport:
    """Run end-to-end MultiAgentGraph evaluation across agent_golden.jsonl."""
    cases: List[Dict] = [
        json.loads(line)
        for line in golden_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    llm_client = llm or RuleBasedMockLLMClient()
    registry, _ = build_default_registry(repo_root=repo_root, log_path=log_path)

    results: List[AgentCaseResult] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        ckpt = SQLiteCheckpointStore(tmp / "eval_state.db")
        mem = DualLayerMemory(tmp / "eval_state.db")
        tracer = TraceRecorder(tmp / "traces")

        for case in cases:
            graph = MultiAgentGraph(
                llm=llm_client,
                registry=registry,
                checkpoint_store=ckpt,
                memory=mem,
                context_manager=ContextManager(),
                tracer=tracer,
            )
            t0 = time.perf_counter()
            state = await graph.run(query=case["prompt"], task_id=f"eval_{case['id']}")
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

            answer_low = state.final_answer.lower()
            expected_file = case["expected_file"].lower()
            file_grounded = expected_file in answer_low

            req_keywords = case.get("expected_keywords", [])
            kw_hits = sum(1 for kw in req_keywords if kw.lower() in answer_low)

            used_tools = {rec.tool_name for rec in state.execution_records if rec.ok}
            req_tools = set(case.get("required_tools", []))
            tools_ok = req_tools.issubset(used_tools)

            passed = file_grounded and (kw_hits >= max(1, len(req_keywords) - 1)) and tools_ok
            reason = ""
            if not passed:
                reasons = []
                if not file_grounded:
                    reasons.append(f"missing file {case['expected_file']}")
                if kw_hits < max(1, len(req_keywords) - 1):
                    reasons.append(f"keywords {kw_hits}/{len(req_keywords)}")
                if not tools_ok:
                    reasons.append(f"missing tools {sorted(req_tools - used_tools)}")
                reason = "; ".join(reasons)

            results.append(
                AgentCaseResult(
                    case_id=case["id"],
                    passed=passed,
                    file_grounded=file_grounded,
                    keywords_matched=kw_hits,
                    keywords_total=len(req_keywords),
                    required_tools_used=tools_ok,
                    latency_ms=latency_ms,
                    total_tokens=state.total_tokens,
                    bad_case_reason=reason,
                )
            )

    total = max(1, len(results))
    passed_cases = sum(1 for r in results if r.passed)
    tool_ok_cases = sum(1 for r in results if r.required_tools_used)
    latencies = [r.latency_ms for r in results]
    avg_tokens = round(sum(r.total_tokens for r in results) / total, 1)

    return AgentEvalReport(
        total_cases=len(results),
        passed_cases=passed_cases,
        pass_rate=round(passed_cases / total * 100.0, 1),
        tool_grounding_rate=round(tool_ok_cases / total * 100.0, 1),
        avg_tokens=avg_tokens,
        p50_latency_ms=percentile(latencies, 50),
        p95_latency_ms=percentile(latencies, 95),
        bad_cases=[r for r in results if not r.passed],
        results=results,
    )
