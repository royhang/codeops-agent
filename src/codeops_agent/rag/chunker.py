from __future__ import annotations

import ast
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field


class CodeChunk(BaseModel):
    """Structured unit of code or log content for indexing and retrieval."""

    chunk_id: str
    file_path: str
    symbol_name: str
    kind: str  # function | class | module | log
    start_line: int
    end_line: int
    content: str
    metadata: dict = Field(default_factory=dict)


def chunk_python_file(file_path: Path, repo_root: Path) -> List[CodeChunk]:
    """Parse a Python source file with AST to extract function/class chunks cleanly."""
    rel_path = file_path.relative_to(repo_root).as_posix()
    text = file_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    if not lines:
        return []

    try:
        tree = ast.parse(text)
    except SyntaxError:
        return [
            CodeChunk(
                chunk_id=f"{rel_path}:1-{len(lines)}",
                file_path=rel_path,
                symbol_name=file_path.stem,
                kind="module",
                start_line=1,
                end_line=len(lines),
                content=text,
            )
        ]

    chunks: List[CodeChunk] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = node.lineno
            end = getattr(node, "end_lineno", start)
            snippet = "\n".join(lines[start - 1 : end])
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            chunks.append(
                CodeChunk(
                    chunk_id=f"{rel_path}:{node.name}:{start}-{end}",
                    file_path=rel_path,
                    symbol_name=node.name,
                    kind=kind,
                    start_line=start,
                    end_line=end,
                    content=f"# file: {rel_path} | symbol: {node.name}\n{snippet}",
                )
            )
            # Also index methods inside classes individually for finer recall
            if isinstance(node, ast.ClassDef):
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        c_start = child.lineno
                        c_end = getattr(child, "end_lineno", c_start)
                        c_snippet = "\n".join(lines[c_start - 1 : c_end])
                        chunks.append(
                            CodeChunk(
                                chunk_id=f"{rel_path}:{node.name}.{child.name}:{c_start}-{c_end}",
                                file_path=rel_path,
                                symbol_name=child.name,
                                kind="function",
                                start_line=c_start,
                                end_line=c_end,
                                content=f"# file: {rel_path} | class: {node.name} | method: {child.name}\n{c_snippet}",
                            )
                        )

    if not chunks:
        chunks.append(
            CodeChunk(
                chunk_id=f"{rel_path}:1-{len(lines)}",
                file_path=rel_path,
                symbol_name=file_path.stem,
                kind="module",
                start_line=1,
                end_line=len(lines),
                content=text,
            )
        )
    return chunks


def chunk_logs(log_path: Path, window_size: int = 3) -> List[CodeChunk]:
    """Chunk structured log files by sliding line windows."""
    if not log_path.is_file():
        return []
    lines = [ln for ln in log_path.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
    chunks: List[CodeChunk] = []
    for idx in range(0, len(lines), window_size):
        block = lines[idx : idx + window_size]
        start = idx + 1
        end = idx + len(block)
        chunks.append(
            CodeChunk(
                chunk_id=f"{log_path.name}:{start}-{end}",
                file_path=log_path.name,
                symbol_name="log_window",
                kind="log",
                start_line=start,
                end_line=end,
                content="\n".join(block),
            )
        )
    return chunks


def chunk_repository(repo_root: Path, log_path: Optional[Path] = None) -> List[CodeChunk]:
    """Walk repository Python/Markdown files and optional log file into structured chunks."""
    repo_root = repo_root.resolve()
    chunks: List[CodeChunk] = []
    ignore_dirs = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "node_modules"}

    for path in sorted(repo_root.rglob("*")):
        if any(part in ignore_dirs for part in path.parts):
            continue
        if not path.is_file():
            continue
        if path.suffix == ".py":
            chunks.extend(chunk_python_file(path, repo_root))
        elif path.suffix in {".md", ".txt", ".yaml", ".yml"}:
            rel_path = path.relative_to(repo_root).as_posix()
            text = path.read_text(encoding="utf-8", errors="replace")
            if text.strip():
                chunks.append(
                    CodeChunk(
                        chunk_id=f"{rel_path}:1",
                        file_path=rel_path,
                        symbol_name=path.stem,
                        kind="doc",
                        start_line=1,
                        end_line=len(text.splitlines()),
                        content=text[:2000],
                    )
                )

    if log_path is not None:
        chunks.extend(chunk_logs(log_path))
    return chunks
