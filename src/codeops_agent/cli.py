from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Optional

import uvicorn

from codeops_agent.agent.checkpoint import SQLiteCheckpointStore
from codeops_agent.agent.graph import MultiAgentGraph
from codeops_agent.api.server import build_llm, create_app
from codeops_agent.config import Settings
from codeops_agent.context.manager import ContextManager
from codeops_agent.context.memory import DualLayerMemory
from codeops_agent.eval.runner import run_eval_cli
from codeops_agent.mcp.server import MCPStdioServer
from codeops_agent.observability.tracer import TraceRecorder
from codeops_agent.tools.builtin import build_default_registry


async def _run_once(
    query: str,
    repo: Path,
    logs: Optional[Path],
    task_id: Optional[str],
) -> None:
    cfg = Settings.from_env()
    registry, _ = build_default_registry(
        repo_root=repo,
        log_path=logs,
        sandbox_timeout=cfg.sandbox_timeout_seconds,
    )
    graph = MultiAgentGraph(
        llm=build_llm(cfg),
        registry=registry,
        checkpoint_store=SQLiteCheckpointStore(cfg.state_db),
        memory=DualLayerMemory(cfg.state_db),
        context_manager=ContextManager(
            max_tool_output_chars=cfg.max_tool_output_chars,
            token_budget=cfg.context_token_budget,
        ),
        tracer=TraceRecorder(cfg.trace_dir),
        max_steps=cfg.max_steps,
    )
    state = await graph.run(query=query, task_id=task_id)
    print(f"[Task ID]: {state.task_id} | [Status]: {state.status} | [Tokens]: {state.total_tokens}")
    print("\n--- Plan ---")
    for step in state.plan:
        print(f"{step.step_id}. {step.goal} ({step.suggested_tool})")
    print("\n--- Final Answer ---")
    print(state.final_answer)


def main() -> None:
    parser = argparse.ArgumentParser(prog="codeops-agent", description="CodeOps-Agent CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run a single troubleshooting task")
    p_run.add_argument("query", type=str)
    p_run.add_argument("--repo", type=Path, default=Path("benchmarks/sample_repo"))
    p_run.add_argument("--logs", type=Path, default=Path("benchmarks/sample_logs/app.log"))
    p_run.add_argument("--task-id", type=str, default=None)

    p_serve = sub.add_parser("serve", help="Start FastAPI SSE server + Web UI")
    p_serve.add_argument("--host", type=str, default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--repo", type=Path, default=Path("benchmarks/sample_repo"))
    p_serve.add_argument("--logs", type=Path, default=Path("benchmarks/sample_logs/app.log"))

    p_eval = sub.add_parser("eval", help="Run RAG & E2E Agent benchmark")
    p_eval.add_argument("--repo", type=Path, default=Path("benchmarks/sample_repo"))
    p_eval.add_argument("--logs", type=Path, default=Path("benchmarks/sample_logs/app.log"))
    p_eval.add_argument("--rag-golden", type=Path, default=Path("benchmarks/rag_golden.jsonl"))
    p_eval.add_argument("--agent-golden", type=Path, default=Path("benchmarks/agent_golden.jsonl"))
    p_eval.add_argument("--output-md", type=Path, default=None)

    p_mcp = sub.add_parser("mcp-serve", help="Start JSON-RPC 2.0 MCP server on stdio")
    p_mcp.add_argument("--repo", type=Path, default=Path("benchmarks/sample_repo"))
    p_mcp.add_argument("--logs", type=Path, default=Path("benchmarks/sample_logs/app.log"))

    args = parser.parse_args()
    if args.command == "run":
        asyncio.run(_run_once(args.query, args.repo, args.logs, args.task_id))
    elif args.command == "serve":
        app = create_app(repo_root=args.repo, log_path=args.logs)
        uvicorn.run(app, host=args.host, port=args.port)
    elif args.command == "eval":
        asyncio.run(
            run_eval_cli(
                repo_root=args.repo,
                log_path=args.logs,
                rag_golden=args.rag_golden,
                agent_golden=args.agent_golden,
                output_md=args.output_md,
            )
        )
    elif args.command == "mcp-serve":
        registry, _ = build_default_registry(repo_root=args.repo, log_path=args.logs)
        MCPStdioServer(registry).serve_stdio_sync()


if __name__ == "__main__":
    main()
