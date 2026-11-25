from __future__ import annotations

from typing import Any, Dict, List

from pydantic import ValidationError

from codeops_agent.tools.base import BaseTool, ToolResult


class ToolRegistry:
    """Central registry for local and MCP-mounted tools with Pydantic validation."""

    def __init__(self) -> None:
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def list_names(self) -> List[str]:
        return sorted(self._tools.keys())

    def openai_schemas(self) -> List[Dict[str, Any]]:
        return [self._tools[name].to_openai_schema() for name in sorted(self._tools.keys())]

    def mcp_schemas(self) -> List[Dict[str, Any]]:
        return [self._tools[name].to_mcp_schema() for name in sorted(self._tools.keys())]

    async def execute(self, name: str, raw_arguments: Dict[str, Any]) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            available = ", ".join(self.list_names())
            return ToolResult(
                ok=False,
                output=f"Unknown tool '{name}'.",
                error_hint=f"Choose one of available tools: {available}",
            )

        if "_raw_invalid_json" in raw_arguments:
            return ToolResult(
                ok=False,
                output="Tool arguments were not valid JSON.",
                error_hint="Return a strictly valid JSON object matching inputSchema.",
            )

        try:
            validated_args = tool.args_schema.model_validate(raw_arguments)
        except ValidationError as exc:
            return ToolResult(
                ok=False,
                output=f"Argument validation failed for '{name}': {exc.errors()}",
                error_hint="Fix parameter names/types according to the tool schema and retry.",
            )

        try:
            return await tool.run(validated_args)
        except Exception as exc:  # pragma: no cover - defensive guard
            return ToolResult(
                ok=False,
                output=f"Unhandled error in tool '{name}': {exc}",
                error_hint="Check arguments or try an alternative tool.",
            )
