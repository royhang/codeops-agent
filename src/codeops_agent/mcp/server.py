from __future__ import annotations

import asyncio
import json
import sys
from typing import Any, Dict, Optional, TextIO

from codeops_agent import __version__
from codeops_agent.tools.registry import ToolRegistry


class MCPStdioServer:
    """Lightweight JSON-RPC 2.0 MCP Server over stdio exposing a ToolRegistry."""

    PROTOCOL_VERSION = "2024-11-05"

    def __init__(self, registry: ToolRegistry, server_name: str = "codeops-mcp") -> None:
        self.registry = registry
        self.server_name = server_name

    async def handle_request(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        req_id = request.get("id")
        method = request.get("method", "")
        params = request.get("params") or {}

        # Notifications (no id) do not require a JSON-RPC response
        if req_id is None:
            return None

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": self.PROTOCOL_VERSION,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": self.server_name, "version": __version__},
                },
            }

        if method == "ping":
            return {"jsonrpc": "2.0", "id": req_id, "result": {}}

        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": self.registry.mcp_schemas()},
            }

        if method == "tools/call":
            tool_name = params.get("name", "")
            arguments = params.get("arguments") or {}
            res = await self.registry.execute(tool_name, arguments)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": res.to_model_text()}],
                    "isError": not res.ok,
                },
            }

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }

    def serve_stdio_sync(self, stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout) -> None:
        """Blocking line-delimited JSON-RPC loop suitable for `codeops-agent mcp-serve`."""
        for raw_line in stdin:
            line = raw_line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
            except json.JSONDecodeError as exc:
                err_resp = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": f"Parse error: {exc}"},
                }
                stdout.write(json.dumps(err_resp, ensure_ascii=False) + "\n")
                stdout.flush()
                continue

            resp = asyncio.run(self.handle_request(req))
            if resp is not None:
                stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
                stdout.flush()
