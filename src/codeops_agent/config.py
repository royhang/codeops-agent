from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


def _load_dotenv_file(dotenv_path: Path) -> None:
    """Load simple KEY=VALUE pairs from a .env file without overriding existing env vars."""
    if not dotenv_path.is_file():
        return
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


class Settings(BaseModel):
    """Runtime configuration for CodeOps-Agent."""

    llm_provider: str = Field(default="mock", description="mock or openai")
    api_key: str = Field(default="", description="OpenAI-compatible API key")
    base_url: str = Field(default="https://api.openai.com/v1")
    model: str = Field(default="gpt-4o-mini")
    embedding_model: str = Field(default="")

    max_steps: int = Field(default=8, ge=1, le=30)
    max_tool_output_chars: int = Field(default=2400, ge=200)
    context_token_budget: int = Field(default=12000, ge=1000)
    sandbox_timeout_seconds: int = Field(default=10, ge=1, le=120)

    state_db: str = Field(default=".codeops_state.db")
    trace_dir: str = Field(default=".codeops_traces")

    langfuse_public_key: str = Field(default="")
    langfuse_secret_key: str = Field(default="")
    langfuse_host: str = Field(default="https://cloud.langfuse.com")

    @classmethod
    def from_env(cls, env_file: Optional[Path] = None) -> "Settings":
        if env_file is not None:
            _load_dotenv_file(env_file)
        else:
            default_env = Path.cwd() / ".env"
            _load_dotenv_file(default_env)

        return cls(
            llm_provider=os.getenv("CODEOPS_LLM_PROVIDER", "mock").strip().lower(),
            api_key=os.getenv("CODEOPS_API_KEY", "").strip(),
            base_url=os.getenv("CODEOPS_BASE_URL", "https://api.openai.com/v1").strip().rstrip("/"),
            model=os.getenv("CODEOPS_MODEL", "gpt-4o-mini").strip(),
            embedding_model=os.getenv("CODEOPS_EMBEDDING_MODEL", "").strip(),
            max_steps=int(os.getenv("CODEOPS_MAX_STEPS", "8")),
            max_tool_output_chars=int(os.getenv("CODEOPS_MAX_TOOL_OUTPUT_CHARS", "2400")),
            context_token_budget=int(os.getenv("CODEOPS_CONTEXT_TOKEN_BUDGET", "12000")),
            sandbox_timeout_seconds=int(os.getenv("CODEOPS_SANDBOX_TIMEOUT_SECONDS", "10")),
            state_db=os.getenv("CODEOPS_STATE_DB", ".codeops_state.db"),
            trace_dir=os.getenv("CODEOPS_TRACE_DIR", ".codeops_traces"),
            langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY", "").strip(),
            langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY", "").strip(),
            langfuse_host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com").strip().rstrip("/"),
        )
