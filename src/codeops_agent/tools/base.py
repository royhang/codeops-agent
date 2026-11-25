from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type

from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """Structured tool execution result with self-correction hints for the LLM."""

    ok: bool = True
    output: str = ""
    error_hint: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_model_text(self) -> str:
        if self.ok:
            return self.output
        hint = f"\n[Correction Hint]: {self.error_hint}" if self.error_hint else ""
        return f"[TOOL_ERROR] {self.output}{hint}"


class ToolSpec(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]


class BaseTool(ABC):
    """Abstract tool interface backed by a Pydantic v2 argument schema."""

    name: str
    description: str
    args_schema: Type[BaseModel]

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters=self.args_schema.model_json_schema(),
        )

    def to_openai_schema(self) -> Dict[str, Any]:
        spec = self.spec()
        return {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters,
            },
        }

    def to_mcp_schema(self) -> Dict[str, Any]:
        spec = self.spec()
        return {
            "name": spec.name,
            "description": spec.description,
            "inputSchema": spec.parameters,
        }

    @abstractmethod
    async def run(self, args: BaseModel) -> ToolResult:
        raise NotImplementedError
