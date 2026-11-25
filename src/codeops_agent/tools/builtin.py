from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from codeops_agent.rag.chunker import chunk_repository
from codeops_agent.rag.hybrid import HybridRetriever
from codeops_agent.tools.base import BaseTool, ToolResult
from codeops_agent.tools.registry import ToolRegistry
from codeops_agent.tools.sandbox import SandboxRunner


class SearchCodeArgs(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language query or symbol/error keyword")
    top_k: int = Field(default=5, ge=1, le=15, description="Number of code chunks to return")


class SearchCodeTool(BaseTool):
    name = "search_code"
    description = "Search the repository using Hybrid RAG (BM25 + Vector RRF + Rerank) to find relevant functions/classes."
    args_schema = SearchCodeArgs

    def __init__(self, retriever: HybridRetriever) -> None:
        self.retriever = retriever

    async def run(self, args: SearchCodeArgs) -> ToolResult:  # type: ignore[override]
        hits = self.retriever.search_reranked(args.query, top_k=args.top_k)
        if not hits:
            return ToolResult(ok=True, output="No matching code chunks found.")
        blocks = []
        for idx, hit in enumerate(hits, start=1):
            c = hit.chunk
            blocks.append(
                f"[{idx}] {c.file_path}:{c.start_line}-{c.end_line} "
                f"(symbol={c.symbol_name}, score={hit.score})\n{c.content}"
            )
        return ToolResult(ok=True, output="\n\n".join(blocks), metadata={"hit_count": len(hits)})


class ReadFileArgs(BaseModel):
    path: str = Field(..., min_length=1, description="Relative path inside repository, e.g. app/pricing.py")
    start_line: int = Field(default=1, ge=1)
    end_line: int = Field(default=120, ge=1, le=500)


class ReadFileTool(BaseTool):
    name = "read_file"
    description = "Read a specific line range from a file inside the repository safely."
    args_schema = ReadFileArgs

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root.resolve()

    async def run(self, args: ReadFileArgs) -> ToolResult:  # type: ignore[override]
        candidate = (self.repo_root / args.path).resolve()
        try:
            candidate.relative_to(self.repo_root)
        except ValueError:
            return ToolResult(
                ok=False,
                output=f"Access denied: '{args.path}' escapes repository root.",
                error_hint="Provide a relative path inside the repository (e.g. app/pricing.py).",
            )
        if not candidate.is_file():
            return ToolResult(
                ok=False,
                output=f"File not found: '{args.path}'.",
                error_hint="Run search_code first to verify the exact file path.",
            )

        lines = candidate.read_text(encoding="utf-8", errors="replace").splitlines()
        start = max(1, args.start_line)
        end = min(len(lines), max(start, args.end_line))
        numbered = [f"{ln_no:4d} | {lines[ln_no - 1]}" for ln_no in range(start, end + 1)]
        header = f"File: {args.path} (lines {start}-{end} of {len(lines)})"
        return ToolResult(ok=True, output=header + "\n" + "\n".join(numbered))


class InspectLogsArgs(BaseModel):
    keyword: str = Field(default="ERROR", description="Keyword or service name to filter log lines")
    limit: int = Field(default=10, ge=1, le=50)


class InspectLogsTool(BaseTool):
    name = "inspect_logs"
    description = "Search runtime application logs by keyword or error signature."
    args_schema = InspectLogsArgs

    def __init__(self, log_path: Optional[Path]) -> None:
        self.log_path = log_path.resolve() if log_path else None

    async def run(self, args: InspectLogsArgs) -> ToolResult:  # type: ignore[override]
        if self.log_path is None or not self.log_path.is_file():
            return ToolResult(ok=True, output="No log file configured or file does not exist.")
        lines = self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        kw = args.keyword.lower()
        matched = [ln for ln in lines if kw in ln.lower()]
        if not matched:
            matched = lines[-args.limit :]
            prefix = f"No exact log match for '{args.keyword}'; showing latest {len(matched)} log lines:"
        else:
            matched = matched[-args.limit :]
            prefix = f"Matched {len(matched)} log lines for '{args.keyword}':"
        return ToolResult(ok=True, output=prefix + "\n" + "\n".join(matched))


class GitHistoryArgs(BaseModel):
    path: str = Field(default="", description="Optional relative file path to filter git log")
    limit: int = Field(default=5, ge=1, le=20)


class GitHistoryTool(BaseTool):
    name = "git_history"
    description = "Inspect recent git commits for the repository or a specific file."
    args_schema = GitHistoryArgs

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root.resolve()

    async def run(self, args: GitHistoryArgs) -> ToolResult:  # type: ignore[override]
        cmd = ["git", "log", f"-n{args.limit}", "--oneline"]
        if args.path:
            cmd.extend(["--", args.path])
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(self.repo_root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out_b, err_b = await proc.communicate()
        if proc.returncode != 0:
            err_msg = err_b.decode("utf-8", errors="replace").strip()
            return ToolResult(ok=True, output=f"Git history unavailable ({err_msg or 'not a git repository'}).")
        out = out_b.decode("utf-8", errors="replace").strip()
        return ToolResult(ok=True, output=out or "No commits found.")


class SandboxCommandArgs(BaseModel):
    command: str = Field(..., min_length=1, description="Allowlisted command, e.g. 'pytest -q' or 'ls app'")


class SandboxCommandTool(BaseTool):
    name = "run_sandbox_command"
    description = "Run a safe allowlisted diagnostic or test command in the jailed repository sandbox."
    args_schema = SandboxCommandArgs

    def __init__(self, sandbox: SandboxRunner) -> None:
        self.sandbox = sandbox

    async def run(self, args: SandboxCommandArgs) -> ToolResult:  # type: ignore[override]
        return await self.sandbox.run(args.command)


def build_default_registry(
    repo_root: Path,
    log_path: Optional[Path] = None,
    sandbox_timeout: int = 10,
) -> tuple[ToolRegistry, HybridRetriever]:
    """Create and populate the default ToolRegistry and HybridRetriever for a target repository."""
    repo_root = repo_root.resolve()
    chunks = chunk_repository(repo_root, log_path=None)
    retriever = HybridRetriever(chunks)
    sandbox = SandboxRunner(repo_root=repo_root, timeout_seconds=sandbox_timeout)

    registry = ToolRegistry()
    registry.register(SearchCodeTool(retriever))
    registry.register(ReadFileTool(repo_root))
    registry.register(InspectLogsTool(log_path))
    registry.register(GitHistoryTool(repo_root))
    registry.register(SandboxCommandTool(sandbox))
    return registry, retriever
