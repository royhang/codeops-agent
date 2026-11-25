from __future__ import annotations

import asyncio
import shlex
from pathlib import Path
from typing import Set

from codeops_agent.tools.base import ToolResult


class SandboxRunner:
    """Safely runs allowlisted read-only/test commands inside a jailed repository directory."""

    DEFAULT_ALLOWLIST: Set[str] = {
        "pytest",
        "python",
        "python3",
        "git",
        "ls",
        "grep",
        "rg",
        "head",
        "tail",
        "wc",
        "cat",
    }

    DANGEROUS_PATTERNS = (";", "&&", "||", "|", "`", "$(", ">", "<", "rm ", "chmod ", "curl ", "wget ")

    def __init__(
        self,
        repo_root: Path,
        timeout_seconds: int = 10,
        allowlist: Set[str] | None = None,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.timeout_seconds = timeout_seconds
        self.allowlist = allowlist or set(self.DEFAULT_ALLOWLIST)

    def validate_command(self, command: str) -> str | None:
        stripped = command.strip()
        if not stripped:
            return "Empty command is not allowed."
        for bad in self.DANGEROUS_PATTERNS:
            if bad in stripped:
                return f"Shell operator or forbidden token '{bad}' is blocked in sandbox."
        try:
            parts = shlex.split(stripped)
        except ValueError as exc:
            return f"Invalid command quoting: {exc}"
        if not parts:
            return "Empty command."
        binary = Path(parts[0]).name
        if binary not in self.allowlist:
            return f"Command '{binary}' is not in sandbox allowlist: {sorted(self.allowlist)}"
        if binary == "git":
            allowed_git_sub = {"status", "log", "diff", "show", "grep", "branch"}
            if len(parts) < 2 or parts[1] not in allowed_git_sub:
                return f"Only read-only git subcommands {sorted(allowed_git_sub)} are allowed."
        return None

    async def run(self, command: str) -> ToolResult:
        err = self.validate_command(command)
        if err is not None:
            return ToolResult(
                ok=False,
                output=err,
                error_hint=f"Use a single allowlisted command from {sorted(self.allowlist)} without shell chaining.",
            )

        parts = shlex.split(command.strip())
        try:
            proc = await asyncio.create_subprocess_exec(
                *parts,
                cwd=str(self.repo_root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout_b, stderr_b = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=self.timeout_seconds,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                return ToolResult(
                    ok=False,
                    output=f"Command timed out after {self.timeout_seconds}s.",
                    error_hint="Narrow down the test or search scope so it finishes faster.",
                )

            out = stdout_b.decode("utf-8", errors="replace").strip()
            err_out = stderr_b.decode("utf-8", errors="replace").strip()
            combined = out if not err_out else f"{out}\n[stderr]\n{err_out}".strip()
            return ToolResult(
                ok=(proc.returncode == 0),
                output=combined or f"(exited with code {proc.returncode})",
                error_hint=None if proc.returncode == 0 else f"Command exited with code {proc.returncode}.",
                metadata={"returncode": proc.returncode},
            )
        except FileNotFoundError:
            return ToolResult(
                ok=False,
                output=f"Executable '{parts[0]}' not found in environment.",
                error_hint="Try another installed command such as python3 or ls.",
            )
