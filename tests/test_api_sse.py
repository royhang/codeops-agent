from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from codeops_agent.api.server import create_app
from codeops_agent.config import Settings

REPO_ROOT = Path(__file__).resolve().parents[1] / "benchmarks" / "sample_repo"
LOG_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "sample_logs" / "app.log"


def test_api_health_tasks_and_sse_streaming(tmp_path: Path) -> None:
    cfg = Settings(
        llm_provider="mock",
        state_db=str(tmp_path / "api_state.db"),
        trace_dir=str(tmp_path / "api_traces"),
    )
    app = create_app(repo_root=REPO_ROOT, log_path=LOG_PATH, settings=cfg)
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    # Synchronous task API
    task_resp = client.post(
        "/api/v1/tasks",
        json={"query": "为什么 SessionToken 刚签发就返回 401？", "task_id": "api_t1"},
    )
    assert task_resp.status_code == 200
    body = task_resp.json()
    assert body["status"] == "completed"
    assert "app/auth.py" in body["final_answer"]

    # Checkpoint lookup
    lookup = client.get("/api/v1/tasks/api_t1")
    assert lookup.status_code == 200
    assert lookup.json()["task_id"] == "api_t1"

    # SSE streaming endpoint
    with client.stream(
        "POST",
        "/api/v1/tasks/stream",
        json={"query": "线上报 ZeroDivisionError in calculate_discount_rate"},
    ) as stream_resp:
        assert stream_resp.status_code == 200
        raw_sse = "".join(stream_resp.iter_text())
    assert "event: planner" in raw_sse
    assert "event: tool_call" in raw_sse
    assert "event: reviewer" in raw_sse
    assert "event: final" in raw_sse

    traces = client.get("/api/v1/traces")
    assert traces.status_code == 200
    assert len(traces.json()["traces"]) >= 2
