from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional

from codeops_agent.llm.base import (
    BaseLLMClient,
    LLMResponse,
    Message,
    TokenUsage,
    ToolCall,
    estimate_tokens,
)


class RuleBasedMockLLMClient(BaseLLMClient):
    """Deterministic multi-role LLM simulator for offline CI, demos, and benchmarks.

    Detects Planner, Executor (ReAct tool-calling), and Reviewer roles from the system prompt
    and emits realistic multi-step tool calls and structured diagnoses without external APIs.
    """

    def __init__(self, model_name: str = "mock-codeops-v1") -> None:
        self.model_name = model_name
        self._call_counter = 0

    async def chat(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.1,
        json_mode: bool = False,
    ) -> LLMResponse:
        _ = (temperature, json_mode)
        t0 = time.perf_counter()
        self._call_counter += 1

        sys_text = "\n".join(m.content for m in messages if m.role == "system")
        user_text = "\n".join(m.content for m in messages if m.role == "user")
        tool_msgs = [m for m in messages if m.role == "tool"]

        prompt_tokens = sum(estimate_tokens(m.content) for m in messages)

        if "ROLE: PLANNER" in sys_text:
            content = self._handle_planner(user_text)
            tool_calls: List[ToolCall] = []
        elif "ROLE: REVIEWER" in sys_text:
            content = self._handle_reviewer(user_text)
            tool_calls = []
        elif "ROLE: SUMMARIZER" in sys_text:
            content = self._handle_summarizer(user_text)
            tool_calls = []
        else:
            content, tool_calls = self._handle_executor(user_text, tool_msgs, tools or [])

        completion_tokens = estimate_tokens(content) + sum(
            estimate_tokens(json.dumps(tc.arguments, ensure_ascii=False) + tc.name)
            for tc in tool_calls
        )
        latency_ms = max(1.2, (time.perf_counter() - t0) * 1000.0)
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            model=self.model_name,
            latency_ms=round(latency_ms, 2),
        )

    def _extract_query_text(self, user_text: str) -> str:
        match = re.search(r"Task:\s*(.+)", user_text, flags=re.IGNORECASE)
        return match.group(1).strip() if match else user_text.strip()

    def _guess_search_keywords(self, query: str) -> str:
        lower = query.lower()
        mapping = [
            (["zerodivision", "discount", "pricing", "coupon", "calculate_discount_rate"], "calculate_discount_rate ZeroDivisionError"),
            (["tenant", "cache", "t_beta", "t_alpha", "串号", "缓存"], "TenantCache build_cache_key"),
            (["token", "401", "session", "expires_at", "过期", "auth"], "is_token_valid expires_at"),
            (["order_db_url", "sqlite", "feature_flag", "config", "local_dev"], "load_service_config ORDER_DB_URL FEATURE_FLAG_V2"),
            (["repository", "1.8gb", "sql 注入", "sql注入", "page_size", "orderrepository", "分页"], "OrderRepository list_orders build_filter_sql"),
            (["payment", "charge", "线程池", "timeout", "卡死", "retry"], "PaymentGatewayClient charge"),
        ]
        for triggers, keywords in mapping:
            if any(t in lower for t in triggers):
                return keywords
        tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", query)
        return " ".join(tokens[:4]) if tokens else "error"

    def _guess_log_keyword(self, query: str) -> str:
        lower = query.lower()
        if "cache" in lower or "tenant" in lower or "缓存" in lower:
            return "cache"
        if "token" in lower or "401" in lower or "auth" in lower:
            return "auth"
        if "config" in lower or "sqlite" in lower or "order_db_url" in lower or "feature_flag" in lower:
            return "bootstrap"
        if "repository" in lower or "1.8gb" in lower or "sql 注入" in lower or "sql注入" in lower:
            return "order-repo"
        if "payment" in lower or "charge" in lower or "线程" in lower:
            return "payment-worker"
        return "ERROR"

    def _handle_planner(self, user_text: str) -> str:
        query = self._extract_query_text(user_text)
        steps = [
            {
                "step_id": 1,
                "goal": f"检索与故障相关的运行日志与代码片段：{query[:60]}",
                "suggested_tool": "inspect_logs / search_code",
            },
            {
                "step_id": 2,
                "goal": "精读命中源码文件，定位引发异常的根因代码行与逻辑缺陷",
                "suggested_tool": "read_file",
            },
            {
                "step_id": 3,
                "goal": "结合证据整理根因说明、修复建议与回归验证方案",
                "suggested_tool": "none",
            },
        ]
        return json.dumps({"plan": steps}, ensure_ascii=False)

    def _handle_executor(
        self,
        user_text: str,
        tool_msgs: List[Message],
        tools: List[Dict[str, Any]],
    ) -> tuple[str, List[ToolCall]]:
        available_names = {t.get("function", {}).get("name", "") for t in tools}
        called_names = [m.name or "" for m in tool_msgs]
        query = self._extract_query_text(user_text)

        # Turn 1: Inspect logs + search codebase in parallel
        if not tool_msgs:
            tcs: List[ToolCall] = []
            if "inspect_logs" in available_names:
                tcs.append(
                    ToolCall(
                        id=f"call_{self._call_counter}_logs",
                        name="inspect_logs",
                        arguments={"keyword": self._guess_log_keyword(query), "limit": 5},
                    )
                )
            if "search_code" in available_names:
                tcs.append(
                    ToolCall(
                        id=f"call_{self._call_counter}_search",
                        name="search_code",
                        arguments={"query": self._guess_search_keywords(query), "top_k": 3},
                    )
                )
            if tcs:
                return ("", tcs)

        # Turn 2: Read the top candidate source file discovered in search_code / logs
        if "read_file" in available_names and "read_file" not in called_names:
            candidate_path = self._extract_candidate_file(tool_msgs, query)
            if candidate_path:
                return (
                    "",
                    [
                        ToolCall(
                            id=f"call_{self._call_counter}_read",
                            name="read_file",
                            arguments={"path": candidate_path, "start_line": 1, "end_line": 120},
                        )
                    ],
                )

        # Turn 3: Synthesize grounded diagnosis directly from observed tool outputs
        diagnosis = self._synthesize_diagnosis(query, tool_msgs)
        return (diagnosis, [])

    def _extract_candidate_file(self, tool_msgs: List[Message], query: str) -> str:
        # Prefer explicit file path observed in inspect_logs or top search_code hit
        log_msgs = "\n".join(m.content for m in tool_msgs if m.name == "inspect_logs")
        log_matches = re.findall(r"(app/[a-zA-Z0-9_]+\.py)", log_msgs)
        if log_matches:
            return log_matches[0]

        search_msgs = "\n".join(m.content for m in tool_msgs if m.name == "search_code")
        search_matches = re.findall(r"\[1\]\s+(app/[a-zA-Z0-9_]+\.py)", search_msgs)
        if search_matches:
            return search_matches[0]
        any_matches = re.findall(r"(app/[a-zA-Z0-9_]+\.py)", search_msgs)
        if any_matches:
            return any_matches[0]

        lower = query.lower()
        if "pricing" in lower or "discount" in lower or "zerodivision" in lower:
            return "app/pricing.py"
        if "cache" in lower or "tenant" in lower or "缓存" in lower:
            return "app/cache.py"
        if "auth" in lower or "token" in lower or "401" in lower:
            return "app/auth.py"
        if "repository" in lower or "sql" in lower or "1.8gb" in lower:
            return "app/repository.py"
        if "payment" in lower or "charge" in lower:
            return "app/client.py"
        if "config" in lower or "order_db" in lower or "feature_flag" in lower:
            return "app/config.py"
        return "app/pricing.py"

    def _synthesize_diagnosis(self, query: str, tool_msgs: List[Message]) -> str:
        # Ground diagnosis primarily on the file actually opened by read_file + user query
        read_msgs = "\n".join(m.content for m in tool_msgs if m.name == "read_file")
        focus_text = (read_msgs + "\n" + query).lower()

        if "app/pricing.py" in focus_text or "calculate_discount_rate" in focus_text or "zerodivision" in focus_text:
            return (
                "### 根因定位\n"
                "- **目标文件**：`app/pricing.py`（函数 `calculate_discount_rate`）\n"
                "- **问题原因**：`gross_total = sum(item.unit_price * item.quantity for item in items)` 当订单为空或商品均为 0 元赠品时，`gross_total == 0`，后续直接执行 `coupon_amount / gross_total` 触发 `ZeroDivisionError`。\n"
                "### 修复建议\n"
                "1. 在除法前增加守卫分支：`if gross_total <= 0: return 0.0`。\n"
                "2. 为空订单与零元订单补充单测验证 `compute_payable_amount` 返回 `0.0`。"
            )
        if "app/cache.py" in focus_text or "build_cache_key" in focus_text or "tenant_id" in focus_text:
            return (
                "### 根因定位\n"
                "- **目标文件**：`app/cache.py`（方法 `TenantCache.build_cache_key`）\n"
                "- **问题原因**：方法忽略了 `tenant_id` 参数（`_ = tenant_id`），仅返回 `f\"{resource}:{item_id}\"`，导致不同租户在共享相同 `order_id` 时发生缓存键冲突，出现 `T_BETA` 读取 `T_ALPHA` 订单摘要的越权串号。\n"
                "### 修复建议\n"
                "1. 修改缓存键格式为 `f\"{tenant_id}:{resource}:{item_id}\"`，并在入口校验 `tenant_id` 非空。\n"
                "2. 补充多租户隔离测试确保同 `order_id` 不同 `tenant_id` 互不覆盖。"
            )
        if "app/auth.py" in focus_text or "is_token_valid" in focus_text or "expires_at" in focus_text or "sessiontoken" in focus_text:
            return (
                "### 根因定位\n"
                "- **目标文件**：`app/auth.py`（函数 `is_token_valid`）\n"
                "- **问题原因**：过期校验条件写反为 `current_ts >= token.expires_at`，导致未过期的新鲜 `SessionToken` 返回 `False`（401 拒绝），而已过期令牌反而返回 `True`。\n"
                "### 修复建议\n"
                "1. 将比较逻辑修正为 `token.issued_at <= current_ts <= token.expires_at`。\n"
                "2. 对刚签发、临界过期和已过期三类 token 添加单元测试。"
            )
        if "app/repository.py" in focus_text or "list_orders" in focus_text or "build_filter_sql" in focus_text:
            return (
                "### 根因定位\n"
                "- **目标文件**：`app/repository.py`（`OrderRepository.list_orders` 与 `build_filter_sql`）\n"
                "- **问题原因**：1）`list_orders` 忽略了 `page` 和 `page_size` 分页参数直接全量返回列表，引发 1.8GB 内存飙升；2）`build_filter_sql` 直接将 `status` 字符串拼接进 SQL 语句，存在 SQL 注入风险。\n"
                "### 修复建议\n"
                "1. 在 `list_orders` 中校验 `page >= 1` 与 `1 <= page_size <= 200` 并进行切片 `matched[offset:offset + page_size]`。\n"
                "2. 将 `build_filter_sql` 改为参数化查询占位符（如 `WHERE status = ?`）并单独返回绑定参数。"
            )
        if "app/client.py" in focus_text or "paymentgatewayclient" in focus_text or "while true" in focus_text:
            return (
                "### 根因定位\n"
                "- **目标文件**：`app/client.py`（方法 `PaymentGatewayClient.charge`）\n"
                "- **问题原因**：使用了无超时配置且无上限的 `while True` 死循环重试下游 `/v1/charge`，在网关抖动时持续占用工作线程，最终打满线程池导致雪崩。\n"
                "### 修复建议\n"
                "1. 设置明确请求超时（如 `timeout=3.0`）和最大重试次数（如 `max_retries=3`）+ 指数退避。\n"
                "2. 仅对幂等安全且可重试的瞬时错误重发，超出上限后抛出明确异常并触发熔断。"
            )
        return (
            "### 根因定位\n"
            "- **目标文件**：`app/config.py`（函数 `load_service_config`）\n"
            "- **问题原因**：1）环境变量名拼写错误为 `\"ORDER_DB_URl\"`（末尾为小写 `l`），导致生产环境找不到 `ORDER_DB_URL` 而回退到 `sqlite:///local_dev.db`；2）`bool(os.getenv(\"FEATURE_FLAG_V2\", \"false\"))` 对非空字符串 `\"false\"` 求值恒为 `True`。\n"
            "### 修复建议\n"
            "1. 修正环境变量名为 `\"ORDER_DB_URL\"`。\n"
            "2. 显式解析布尔配置：`os.getenv(\"FEATURE_FLAG_V2\", \"false\").strip().lower() in {\"1\", \"true\", \"yes\", \"on\"}`。"
        )

    def _handle_reviewer(self, user_text: str) -> str:
        has_evidence = ("app/" in user_text) and ("根因" in user_text or "root cause" in user_text.lower() or "修复" in user_text)
        payload = {
            "approved": bool(has_evidence),
            "confidence": 0.94 if has_evidence else 0.4,
            "critique": "诊断结论已引用具体源码文件与函数证据，修复建议可执行。" if has_evidence else "缺少具体源码文件路径或证据支撑，请先检索并精读相关代码。",
        }
        return json.dumps(payload, ensure_ascii=False)

    def _handle_summarizer(self, user_text: str) -> str:
        lines = [ln.strip() for ln in user_text.splitlines() if ln.strip()]
        kept = lines[:4]
        return "[历史摘要] " + " | ".join(item[:80] for item in kept)
