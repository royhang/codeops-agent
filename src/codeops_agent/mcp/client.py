from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, create_model

from codeops_agent.tools.base import BaseTool, ToolResult
from codeops_agent.tools.registry import ToolRegistry


class _DynamicMCPTool(BaseTool):
    """Wraps a remote MCP tool as a native BaseTool inside ToolRegistry."""

    def __init__(
        self,
        name: str,
        description: str,
        input_schema: Dict[str, Any],
        client: "MCPStdioClient",
    ) -> None:
        self.name = name
        self.description = description
        self._input_schema = input_schema
        self._client = client

        fields: Dict[str, Any] = {}
        props = input_schema.get("properties") or {}
        required = set(input_schema.get("required") or [])
        for prop_name, prop_meta in props.items():
            py_type: Any = str
            p_type = prop_meta.get("type")
            if p_type == "integer":
                py_type = int
            elif p_type == "number":
                py_type = float
            elif p_type == "boolean":
                py_type = bool
            default = ... if prop_name in required else prop_meta.get("default", None)
            fields[prop_name] = (py_type if prop_name in required else Optional[py_type], default)

        self.args_schema = create_model(f"MCPArgs_{name}", **fields)

    def to_mcp_schema(self) -> Dict[str, Any]:
        return {"name": self.name, "description": self.description, "inputSchema": self._input_schema}

    async def run(self, args: BaseModel) -> ToolResult:
        payload = {k: v for k, v in args.model_dump().items() if v is not None}
        return await self._client.call_tool(self.name, payload)


class MCPStdioClient:
    """Async JSON-RPC 2.0 MCP Client communicating with an MCP Server over subprocess stdio."""

    def __init__(self, command: List[str], cwd: Optional[Path] = None) -> None:
        self.command = command
        self.cwd = cwd
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._req_id = 0

    @classmethod
    def for_codeops_server(cls, repo_root: Path, log_path: Optional[Path] = None) -> "MCPStdioClient":
        cmd = [sys.executable, "-m", "codeops_agent.cli", "mcp-serve", "--repo", str(repo_root)]
        if log_path is not None:
            cmd.extend(["--logs", str(log_path)])
        return cls(command=cmd)

    async def __aenter__(self) -> "MCPStdioClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.close()

    async def connect(self) -> Dict[str, Any]:
        self._proc = await asyncio.create_subprocess_exec(
            *self.command,
            cwd=str(self.cwd) if self.cwd else None,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        init_res = await self._rpc(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "codeops-client", "version": "0.1.0"},
            },
        )
        await self._notify("notifications/initialized", {})
        return init_res

    async def close(self) -> None:
        if self._proc is not None:
            if self._proc.stdin and not self._proc.stdin.is_closing():
                self._proc.stdin.close()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                self._proc.kill()
                await self._proc.wait()
            self._proc = None

    async def list_tools(self) -> List[Dict[str, Any]]:
        res = await self._rpc("tools/list", {})
        return list(res.get("tools") or [])

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> ToolResult:
        res = await self._rpc("tools/call", {"name": name, "arguments": arguments})
        is_err = bool(res.get("isError", False))
        blocks = res.get("content") or []
        text = "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        return ToolResult(ok=not is_err, output=text)

    async def mount_into_registry(self, registry: ToolRegistry) -> List[str]:
        tools = await self.list_tools()
        mounted: List[str] = []
        for meta in tools:
            tool = _DynamicMCPTool(
                name=meta["name"],
                description=meta.get("description", ""),
                input_schema=meta.get("inputSchema") or {"type": "object", "properties": {}},
                client=self,
            )
            registry.register(tool)
            mounted.append(tool.name)
        return mounted

    async def _notify(self, method: str, params: Dict[str, Any]) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("MCP client not connected")
        msg = json.dumps({"jsonrpc": "2.0", "method": method, "params": params}, ensure_ascii=False) + "\n"
        self._proc.stdin.write(msg.encode("utf-8"))
        await self._proc.stdin.drain()

    async def _rpc(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if self._proc is None or self._proc.stdin is None or self._proc.stdout is None:
            raise RuntimeError("MCP client not connected")
        self._req_id += 1
        req_id = self._req_id
        payload = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params,
        }
        line = json.dumps(payload, ensure_ascii=False) + "\n"
        self._proc.stdin.write(line.encode("utf-8"))
        await self._proc.stdin.drain()

        raw_resp = await asyncio.wait_for(self._proc.stdout.readline(), timeout=10.0)
        if not raw_resp:
            stderr_txt = ""
            if self._proc.stderr:
                stderr_txt = (await self._proc.stderr.read()).decode("utf-8", errors="replace")
            raise RuntimeError(f"MCP server closed stream unexpectedly: {stderr_txt}")

        data = json.loads(raw_resp.decode("utf-8"))
        if "error" in data:
            raise RuntimeError(f"MCP RPC error: {data['error']}")
        return dict(data.get("result") or {})
