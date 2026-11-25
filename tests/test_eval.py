from __future__ import annotations

from pathlib import Path

import pytest

from codeops_agent.eval.agent_eval import evaluate_agent
from codeops_agent.eval.runner import run_eval_cli

BENCH_DIR = Path(__file__).resolve().parents[1] / "benchmarks"


@pytest.mark.asyncio
async def test_agent_golden_eval_passes() -> None:
    report = await evaluate_agent(
        repo_root=BENCH_DIR / "sample_repo",
        log_path=BENCH_DIR / "sample_logs" / "app.log",
        golden_path=BENCH_DIR / "agent_golden.jsonl",
    )
    assert report.total_cases == 6
    assert report.pass_rate == 100.0
    assert report.tool_grounding_rate == 100.0


@pytest.mark.asyncio
async def test_eval_cli_writes_markdown_report(tmp_path: Path) -> None:
    out_md = tmp_path / "eval_report.md"
    rag_summary, agent_report = await run_eval_cli(
        repo_root=BENCH_DIR / "sample_repo",
        log_path=BENCH_DIR / "sample_logs" / "app.log",
        rag_golden=BENCH_DIR / "rag_golden.jsonl",
        agent_golden=BENCH_DIR / "agent_golden.jsonl",
        output_md=out_md,
    )
    assert out_md.is_file()
    text = out_md.read_text(encoding="utf-8")
    assert "Hybrid RAG Retrieval Benchmark" in text
    assert "End-to-End Multi-Agent Troubleshooting Benchmark" in text
    assert rag_summary.total_queries == 12
    assert agent_report.passed_cases == 6
