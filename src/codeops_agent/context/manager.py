from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel

from codeops_agent.llm.base import BaseLLMClient, Message, estimate_tokens


class ContextStats(BaseModel):
    raw_chars: int = 0
    kept_chars: int = 0
    saved_chars: int = 0
    saved_tokens_est: int = 0
    compression_events: int = 0


class ContextManager:
    """Manages prompt context budgets via head/tail tool truncation and rolling history compression."""

    def __init__(
        self,
        max_tool_output_chars: int = 2400,
        token_budget: int = 12000,
        keep_recent_messages: int = 6,
    ) -> None:
        self.max_tool_output_chars = max_tool_output_chars
        self.token_budget = token_budget
        self.keep_recent_messages = keep_recent_messages
        self.stats = ContextStats()

    def truncate_tool_output(self, text: str) -> str:
        """Preserve head and tail of verbose tool output (logs/files) while dropping middle noise."""
        self.stats.raw_chars += len(text)
        if len(text) <= self.max_tool_output_chars:
            self.stats.kept_chars += len(text)
            return text

        head_len = int(self.max_tool_output_chars * 0.6)
        tail_len = self.max_tool_output_chars - head_len
        omitted = len(text) - (head_len + tail_len)
        truncated = (
            f"{text[:head_len]}\n"
            f"... [ContextManager truncated {omitted} middle chars to protect token budget] ...\n"
            f"{text[-tail_len:]}"
        )
        self.stats.kept_chars += len(truncated)
        saved = max(0, len(text) - len(truncated))
        self.stats.saved_chars += saved
        self.stats.saved_tokens_est += estimate_tokens("x" * saved)
        self.stats.compression_events += 1
        return truncated

    async def compact_messages(
        self,
        messages: List[Message],
        llm: Optional[BaseLLMClient] = None,
    ) -> List[Message]:
        """Compress middle conversation turns into a rolling summary when exceeding token budget."""
        total_tokens = sum(estimate_tokens(m.content) for m in messages)
        if total_tokens <= self.token_budget or len(messages) <= self.keep_recent_messages + 2:
            return messages

        system_msgs = [m for m in messages if m.role == "system"]
        non_system = [m for m in messages if m.role != "system"]
        if len(non_system) <= self.keep_recent_messages + 1:
            return messages

        # Always keep the initial user task message + most recent N turns
        first_user = non_system[0]
        middle = non_system[1 : -self.keep_recent_messages]
        recent = non_system[-self.keep_recent_messages :]

        middle_text = "\n".join(f"[{m.role}] {m.content[:300]}" for m in middle)
        if llm is not None:
            resp = await llm.chat(
                [
                    Message(role="system", content="ROLE: SUMMARIZER. Summarize key tool findings concisely."),
                    Message(role="user", content=middle_text),
                ]
            )
            summary_body = resp.content.strip()
        else:
            summary_body = " | ".join(
                f"{m.role}:{m.content.splitlines()[0][:80]}" for m in middle if m.content.strip()
            )

        summary_msg = Message(
            role="system",
            content=f"[Rolling Context Summary of {len(middle)} earlier turns]\n{summary_body}",
        )
        compacted = [*system_msgs, first_user, summary_msg, *recent]

        before_tokens = total_tokens
        after_tokens = sum(estimate_tokens(m.content) for m in compacted)
        if after_tokens < before_tokens:
            self.stats.saved_tokens_est += before_tokens - after_tokens
            self.stats.compression_events += 1
        return compacted
