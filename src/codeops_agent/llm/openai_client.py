from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, List, Optional

import httpx

from codeops_agent.llm.base import (
    BaseLLMClient,
    LLMResponse,
    Message,
    TokenUsage,
    ToolCall,
    estimate_tokens,
)


class OpenAICompatibleClient(BaseLLMClient):
    """Async OpenAI-compatible client supporting retries and stable prompt prefix ordering."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        timeout_seconds: float = 45.0,
        max_retries: int = 2,
    ) -> None:
        if not api_key:
            raise ValueError("CODEOPS_API_KEY is required when llm_provider=openai")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    async def chat(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.1,
        json_mode: bool = False,
    ) -> LLMResponse:
        # Keep system messages at the beginning and tools sorted deterministically
        # to maximize provider KV Cache prefix reuse across agent loop turns.
        ordered_messages = [m.to_openai_dict() for m in messages]
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": ordered_messages,
            "temperature": temperature,
        }
        if tools:
            body["tools"] = sorted(tools, key=lambda t: t.get("function", {}).get("name", ""))
            body["tool_choice"] = "auto"
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}/chat/completions"

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            t0 = time.perf_counter()
            try:
                async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                    resp = await client.post(url, headers=headers, json=body)
                    resp.raise_for_status()
                    data = resp.json()
                latency_ms = (time.perf_counter() - t0) * 1000.0
                return self._parse_response(data, ordered_messages, latency_ms)
            except (httpx.HTTPError, ValueError) as exc:
                last_err = exc
                if attempt < self.max_retries:
                    await asyncio.sleep(0.5 * (2**attempt))

        raise RuntimeError(f"OpenAICompatibleClient failed after retries: {last_err}")

    def _parse_response(
        self,
        data: Dict[str, Any],
        ordered_messages: List[Dict[str, Any]],
        latency_ms: float,
    ) -> LLMResponse:
        choice = data["choices"][0]
        msg = choice["message"]
        content = msg.get("content") or ""
        raw_tcs = msg.get("tool_calls") or []
        tool_calls: List[ToolCall] = []
        for idx, raw_tc in enumerate(raw_tcs):
            fn = raw_tc.get("function", {})
            raw_args = fn.get("arguments", "{}")
            try:
                parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
            except json.JSONDecodeError:
                parsed_args = {"_raw_invalid_json": raw_args}
            tool_calls.append(
                ToolCall(
                    id=raw_tc.get("id", f"call_{idx}"),
                    name=fn.get("name", ""),
                    arguments=parsed_args,
                )
            )

        raw_usage = data.get("usage") or {}
        if raw_usage:
            usage = TokenUsage(
                prompt_tokens=int(raw_usage.get("prompt_tokens", 0)),
                completion_tokens=int(raw_usage.get("completion_tokens", 0)),
                total_tokens=int(raw_usage.get("total_tokens", 0)),
            )
        else:
            p_tok = sum(estimate_tokens(json.dumps(m, ensure_ascii=False)) for m in ordered_messages)
            c_tok = estimate_tokens(content)
            usage = TokenUsage(prompt_tokens=p_tok, completion_tokens=c_tok, total_tokens=p_tok + c_tok)

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            model=data.get("model", self.model),
            latency_ms=round(latency_ms, 2),
        )
