from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """Normalized representation of a model tool invocation."""

    id: str
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    """Unified chat message compatible with OpenAI tool-calling format."""

    role: str  # system | user | assistant | tool
    content: str = ""
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: List[ToolCall] = Field(default_factory=list)

    def to_openai_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name:
            payload["name"] = self.name
        if self.tool_call_id:
            payload["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            import json

            payload["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                    },
                }
                for tc in self.tool_calls
            ]
        return payload


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMResponse(BaseModel):
    content: str = ""
    tool_calls: List[ToolCall] = Field(default_factory=list)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    model: str = "unknown"
    latency_ms: float = 0.0


def estimate_tokens(text: str) -> int:
    """Lightweight cross-language token estimator (approx 3 chars/token for mixed CN/EN/code)."""
    if not text:
        return 0
    return max(1, (len(text) + 2) // 3)


class BaseLLMClient(ABC):
    """Abstract async LLM interface."""

    @abstractmethod
    async def chat(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.1,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Send chat completion request and return normalized response."""
        raise NotImplementedError
