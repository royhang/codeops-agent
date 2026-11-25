# CodeOps-Agent 源码精读与动手改造指南

这份文档帮你**在 1~2 周内彻底吃透项目每一行代码**，并知道**如何在不依赖参考答案的情况下自己动手改写、接入真实大模型 API、跑出自己的简历数据、从容应对大厂 Agent 面试深挖**。

---

## 第一步：先把项目在本地跑起来（15 分钟）

```bash
cd codeops-agent
source .venv/bin/activate

# 1. 运行全部单元测试
pytest -v

# 2. 运行离线 RAG + Multi-Agent 评测脚本
codeops-agent eval

# 3. 运行一次命令行排障
codeops-agent run "多租户出现缓存串号，T_BETA 读到了 T_ALPHA 的订单摘要"

# 4. 启动 Web UI (FastAPI + SSE 实时轨迹)
codeops-agent serve --port 8000
# 浏览器打开 http://127.0.0.1:8000
```

---

## 第二步：按模块精读源码（推荐阅读顺序）

不要一上来乱看文件，按 **数据流从底层到编排层** 的顺序读，一共 7 个核心文件：

### 1. 工具抽象与 Pydantic 强校验 (`src/codeops_agent/tools/`)
- 必读文件：[base.py](../src/codeops_agent/tools/base.py)、[registry.py](../src/codeops_agent/tools/registry.py)、[sandbox.py](../src/codeops_agent/tools/sandbox.py)
- 重点看：
  1. `ToolResult.to_model_text()`：为什么报错时要返回 `error_hint`？（引导模型自我纠错，而不是直接抛异常崩溃）
  2. `ReadFileTool` 中的 `candidate.relative_to(self.repo_root)`：如何防御 `../../etc/passwd` 路径穿越攻击。
  3. `SandboxRunner.validate_command`：白名单 + 禁用 `; && | $(` + 超时 `asyncio.wait_for` + `proc.kill()`。
- **动手改造任务 1**：在 `builtin.py` 里新增一个工具 `search_symbol_references(symbol: str)`（统计某个函数在哪些文件被调用），并给它写单测。

### 2. Hybrid RAG 检索链路 (`src/codeops_agent/rag/`)
- 必读文件：[chunker.py](../src/codeops_agent/rag/chunker.py)、[bm25.py](../src/codeops_agent/rag/bm25.py)、[vector.py](../src/codeops_agent/rag/vector.py)、[hybrid.py](../src/codeops_agent/rag/hybrid.py)
- 重点看：
  1. 为什么代码检索不能用“固定 500 字符切分”？看 `chunk_python_file` 如何用 Python `ast` 按函数/类边界切片。
  2. `code_tokenize` 为什么要拆分 `camelCase` 和 `snake_case`？
  3. `HybridRetriever.search_hybrid` 中的 **RRF（Reciprocal Rank Fusion）** 公式：`1 / (60 + rank)`，为什么比直接把 BM25 分数和 Cosine 分数相加更稳？
- **动手改造任务 2**：在 `vector.py` 中新增一个 `OpenAICompatibleEmbedder(BaseEmbedder)`（或接入本地 `sentence-transformers` BGE 模型），并在 `benchmarks/rag_golden.jsonl` 基础上再增加 10 条更口语化的中文查询，对比离线 Hash Vector 与真实 Embedding 的 Recall@5 差异。

### 3. 上下文工程与双层记忆 (`src/codeops_agent/context/`)
- 必读文件：[manager.py](../src/codeops_agent/context/manager.py)、[memory.py](../src/codeops_agent/context/memory.py)
- 重点看：
  1. `truncate_tool_output`：为什么采用 **60% 头部 + 40% 尾部** 保留，丢弃中间部分？（因为堆栈报错信息通常在开头或末尾）
  2. `compact_messages`：如何在超预算时保留 `system` + 首条 `user` 任务 + 最近 N 轮对话，只压缩中间历史？
  3. `DualLayerMemory`：短期滑动窗口 vs SQLite 持久化长期记忆。
- **动手改造任务 3**：把 `DualLayerMemory.recall_long_term` 从关键词命中升级为结合 `LocalHashEmbedder` 的余弦相似度检索。

### 4. 无框架原生 ReAct Loop (`src/codeops_agent/agent/react.py`)
- 必读文件：[react.py](../src/codeops_agent/agent/react.py)
- 重点看：
  1. `asyncio.gather` 如何并发执行同一轮的多个 Tool Call（比如同时查日志 + 搜代码）。
  2. `max_steps` 熔断器：如何防止模型死循环调用工具把 Token 烧干。
- **动手改造任务 4（面试高频）**：试着不看 `react.py`，自己在草稿文件里重新默写一个 60 行的 `ReAct` 循环，把 `OpenAICompatibleClient` 的 `tools` 参数传递与 `tool_calls` 回填逻辑走一遍。

### 5. Multi-Agent 状态机、Checkpoint 与 HITL (`src/codeops_agent/agent/graph.py`)
- 必读文件：[state.py](../src/codeops_agent/agent/state.py)、[checkpoint.py](../src/codeops_agent/agent/checkpoint.py)、[graph.py](../src/codeops_agent/agent/graph.py)
- 重点看：
  1. `Planner -> Executor -> Reviewer` 三节点协作与退回重做（`critique_hint`）。
  2. `SQLiteCheckpointStore` 如何把每一步的 `AgentState` 序列化保存，并在 `HITL` 人工审批后恢复运行。

### 6. 标准 MCP Server / Client (`src/codeops_agent/mcp/`)
- 必读文件：[server.py](../src/codeops_agent/mcp/server.py)、[client.py](../src/codeops_agent/mcp/client.py)
- 重点看：JSON-RPC 2.0 `initialize` -> `notifications/initialized` -> `tools/list` -> `tools/call` 四步握手与调用协议。
- **动手改造任务 5**：把你自己的项目连接成 MCP Client，或用 Cursor / Claude Desktop / Antigravity 挂载 `codeops-agent mcp-serve --repo ...`。

### 7. FastAPI SSE 与自动化评测 (`src/codeops_agent/api/`, `src/codeops_agent/eval/`)
- 必读文件：[server.py](../src/codeops_agent/api/server.py)、[rag_eval.py](../src/codeops_agent/eval/rag_eval.py)、[agent_eval.py](../src/codeops_agent/eval/agent_eval.py)

---

## 第三步：接入真实大模型（DeepSeek / 通义千问 / OpenAI）

项目默认使用 `CODEOPS_LLM_PROVIDER=mock`（方便零门槛离线验证和 CI）。当你申请了 DeepSeek 或阿里云百炼 API Key 后：

```bash
cp .env.example .env
# 编辑 .env：
# CODEOPS_LLM_PROVIDER=openai
# CODEOPS_API_KEY=sk-你的key
# CODEOPS_BASE_URL=https://api.deepseek.com/v1
# CODEOPS_MODEL=deepseek-chat
```

然后重新运行评测：
```bash
codeops-agent eval --output-md eval_reports/deepseek_eval.md
```

**建议你在寒假期间做一次真实的 Bad Case 迭代实验**（面试时最加分）：
1. 先找一个你自己熟悉的开源小项目（或者扩大 `benchmarks/sample_repo` 到 20+ 文件、40+ 条问题），用真实模型跑一遍 `eval` 记录基线准确率（比如 65%）。
2. 查看 `.codeops_traces/traces.jsonl`，找出失败原因（例如：某个工具描述不清导致传参错误、代码分块太碎、长日志把上下文挤爆）。
3. 针对性优化后重跑 `eval`，记录优化后的成功率、Token 降幅与 P95 延迟——**把这组真实实验数据填进你的简历和 GitHub README 中！**
