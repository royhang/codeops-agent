from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from codeops_agent.agent.checkpoint import SQLiteCheckpointStore
from codeops_agent.agent.graph import MultiAgentGraph
from codeops_agent.config import Settings
from codeops_agent.context.manager import ContextManager
from codeops_agent.context.memory import DualLayerMemory
from codeops_agent.llm.base import BaseLLMClient
from codeops_agent.llm.mock_client import RuleBasedMockLLMClient
from codeops_agent.llm.openai_client import OpenAICompatibleClient
from codeops_agent.observability.tracer import TraceRecorder
from codeops_agent.tools.builtin import build_default_registry


class TaskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Troubleshooting task or bug description")
    task_id: Optional[str] = None


def build_llm(settings: Settings) -> BaseLLMClient:
    if settings.llm_provider == "openai" and settings.api_key:
        return OpenAICompatibleClient(
            api_key=settings.api_key,
            base_url=settings.base_url,
            model=settings.model,
        )
    return RuleBasedMockLLMClient()


def create_app(
    repo_root: Path | str = "benchmarks/sample_repo",
    log_path: Optional[Path | str] = "benchmarks/sample_logs/app.log",
    settings: Optional[Settings] = None,
) -> FastAPI:
    cfg = settings or Settings.from_env()
    repo_p = Path(repo_root).resolve()
    log_p = Path(log_path).resolve() if log_path else None

    registry, _ = build_default_registry(
        repo_root=repo_p,
        log_path=log_p,
        sandbox_timeout=cfg.sandbox_timeout_seconds,
    )
    checkpoint_store = SQLiteCheckpointStore(cfg.state_db)
    memory = DualLayerMemory(cfg.state_db)
    tracer = TraceRecorder(cfg.trace_dir)
    llm = build_llm(cfg)

    def _new_graph() -> MultiAgentGraph:
        return MultiAgentGraph(
            llm=llm,
            registry=registry,
            checkpoint_store=checkpoint_store,
            memory=memory,
            context_manager=ContextManager(
                max_tool_output_chars=cfg.max_tool_output_chars,
                token_budget=cfg.context_token_budget,
            ),
            tracer=tracer,
            max_steps=cfg.max_steps,
        )

    app = FastAPI(title="CodeOps-Agent API", version="0.1.0")
    static_index = Path(__file__).parent / "static" / "index.html"

    @app.get("/health")
    async def health() -> dict:
        return {
            "status": "ok",
            "provider": cfg.llm_provider,
            "repo": str(repo_p),
            "tools": registry.list_names(),
        }

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        if not static_index.is_file():
            return "<h1>CodeOps-Agent API</h1><p>Use /docs for OpenAPI.</p>"
        return static_index.read_text(encoding="utf-8")

    @app.post("/api/v1/tasks")
    async def run_task(req: TaskRequest) -> dict:
        graph = _new_graph()
        state = await graph.run(query=req.query, task_id=req.task_id)
        return state.model_dump()

    @app.post("/api/v1/tasks/stream")
    async def stream_task(req: TaskRequest) -> StreamingResponse:
        graph = _new_graph()

        async def _sse_gen():
            async for event in graph.stream(query=req.query, task_id=req.task_id):
                payload = json.dumps(event.model_dump(), ensure_ascii=False)
                yield f"event: {event.event_type}\ndata: {payload}\n\n"

        return StreamingResponse(_sse_gen(), media_type="text/event-stream")

    @app.get("/api/v1/tasks/{task_id}")
    async def get_task(task_id: str) -> dict:
        state = checkpoint_store.load(task_id)
        if state is None:
            raise HTTPException(status_code=404, detail="Task not found")
        return state.model_dump()

    @app.get("/api/v1/traces")
    async def list_traces(limit: int = 20) -> dict:
        return {"traces": tracer.list_recent_traces(limit=limit)}

    return app
