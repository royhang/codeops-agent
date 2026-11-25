from __future__ import annotations

from pathlib import Path

import pytest

from codeops_agent.mcp.client import MCPStdioClient
from codeops_agent.tools.builtin import build_default_registry
from codeops_agent.tools.registry import ToolRegistry
from codeops_agent.tools.sandbox import SandboxRunner

REPO_ROOT = Path(__file__).resolve().parents[1] / "benchmarks" / "sample_repo"
LOG_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "sample_logs" / "app.log"


@pytest.mark.asyncio
async def test_tool_validation_and_path_traversal_guard() -> None:
    registry, _ = build_default_registry(REPO_ROOT, LOG_PATH)

    # Path traversal must be rejected with correction hint
    res = await registry.execute("read_file", {"path": "../../etc/passwd"})
    assert not res.ok
    assert "escapes repository root" in res.output
    assert res.error_hint is not None

    # Invalid Pydantic argument type/missing required field returns structured hint
    bad_args = await registry.execute("search_code", {"top_k": 3})
    assert not bad_args.ok
    assert "validation failed" in bad_args.output.lower()
    assert bad_args.error_hint is not None


@pytest.mark.asyncio
async def test_sandbox_blocks_shell_injection_and_allows_safe_commands() -> None:
    sandbox = SandboxRunner(repo_root=REPO_ROOT, timeout_seconds=5)

    blocked = await sandbox.run("ls && rm -rf /")
    assert not blocked.ok
    assert "blocked" in blocked.output.lower()

    disallowed = await sandbox.run("curl https://example.com")
    assert not disallowed.ok

    ok_res = await sandbox.run("ls app")
    assert ok_res.ok
    assert "pricing.py" in ok_res.output


@pytest.mark.asyncio
async def test_mcp_stdio_roundtrip_and_mount() -> None:
    async with MCPStdioClient.for_codeops_server(REPO_ROOT, LOG_PATH) as client:
        tools = await client.list_tools()
        names = {t["name"] for t in tools}
        assert {"search_code", "read_file", "inspect_logs", "run_sandbox_command"}.issubset(names)

        res = await client.call_tool("read_file", {"path": "app/auth.py", "start_line": 1, "end_line": 25})
        assert res.ok
        assert "is_token_valid" in res.output

        mounted_registry = ToolRegistry()
        mounted_names = await client.mount_into_registry(mounted_registry)
        assert "search_code" in mounted_names

        via_registry = await mounted_registry.execute("inspect_logs", {"keyword": "auth", "limit": 3})
        assert via_registry.ok
        assert "app/auth.py" in via_registry.output
