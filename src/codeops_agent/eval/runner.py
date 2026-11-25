from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Optional

from codeops_agent.config import Settings
from codeops_agent.eval.agent_eval import AgentEvalReport, evaluate_agent
from codeops_agent.eval.rag_eval import RagEvalSummary, evaluate_rag
from codeops_agent.llm.mock_client import RuleBasedMockLLMClient
from codeops_agent.llm.openai_client import OpenAICompatibleClient


def format_markdown_report(rag: RagEvalSummary, agent: AgentEvalReport, provider: str) -> str:
    lines = [
        f"# CodeOps-Agent Evaluation Report (`provider={provider}`)",
        "",
        f"## 1. Hybrid RAG Retrieval Benchmark ({rag.total_queries} Golden Queries)",
        "",
        "| Strategy | Recall@1 | Recall@3 | Recall@5 | MRR@5 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for s in rag.strategies:
        lines.append(
            f"| {s.strategy} | {s.recall_at_1:.1f}% | {s.recall_at_3:.1f}% | {s.recall_at_5:.1f}% | {s.mrr_at_5:.4f} |"
        )

    lines.extend(
        [
            "",
            f"## 2. End-to-End Multi-Agent Troubleshooting Benchmark ({agent.total_cases} Tasks)",
            "",
            f"- **Pass Rate**: {agent.pass_rate:.1f}% ({agent.passed_cases}/{agent.total_cases})",
            f"- **Tool Grounding Rate**: {agent.tool_grounding_rate:.1f}%",
            f"- **Avg Tokens / Task**: {agent.avg_tokens}",
            f"- **Latency (P50 / P95)**: {agent.p50_latency_ms:.2f} ms / {agent.p95_latency_ms:.2f} ms",
            f"- **Bad Cases**: {len(agent.bad_cases)}",
            "",
            "| Case | Passed | File Grounded | Keywords | Tools OK | Latency (ms) | Tokens |",
            "| --- | :---: | :---: | :---: | :---: | ---: | ---: |",
        ]
    )
    for r in agent.results:
        lines.append(
            f"| {r.case_id} | {'PASS' if r.passed else 'FAIL'} | "
            f"{'YES' if r.file_grounded else 'NO'} | {r.keywords_matched}/{r.keywords_total} | "
            f"{'YES' if r.required_tools_used else 'NO'} | {r.latency_ms:.2f} | {r.total_tokens} |"
        )
    return "\n".join(lines) + "\n"


async def run_eval_cli(
    repo_root: Path,
    log_path: Optional[Path],
    rag_golden: Path,
    agent_golden: Path,
    output_md: Optional[Path] = None,
) -> tuple[RagEvalSummary, AgentEvalReport]:
    cfg = Settings.from_env()
    if cfg.llm_provider == "openai" and cfg.api_key:
        llm = OpenAICompatibleClient(
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            model=cfg.model,
        )
        provider_label = f"openai:{cfg.model}"
    else:
        llm = RuleBasedMockLLMClient()
        provider_label = "mock-offline"

    rag_summary = evaluate_rag(repo_root=repo_root, golden_path=rag_golden)
    agent_report = await evaluate_agent(
        repo_root=repo_root,
        log_path=log_path,
        golden_path=agent_golden,
        llm=llm,
    )
    md = format_markdown_report(rag_summary, agent_report, provider_label)
    print(md)
    if output_md is not None:
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(md, encoding="utf-8")
    return rag_summary, agent_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CodeOps-Agent RAG & E2E benchmarks.")
    parser.add_argument("--repo", type=Path, default=Path("benchmarks/sample_repo"))
    parser.add_argument("--logs", type=Path, default=Path("benchmarks/sample_logs/app.log"))
    parser.add_argument("--rag-golden", type=Path, default=Path("benchmarks/rag_golden.jsonl"))
    parser.add_argument("--agent-golden", type=Path, default=Path("benchmarks/agent_golden.jsonl"))
    parser.add_argument("--output-md", type=Path, default=None)
    args = parser.parse_args()

    asyncio.run(
        run_eval_cli(
            repo_root=args.repo,
            log_path=args.logs,
            rag_golden=args.rag_golden,
            agent_golden=args.agent_golden,
            output_md=args.output_md,
        )
    )


if __name__ == "__main__":
    main()
