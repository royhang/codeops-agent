FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY pyproject.toml README.md ./
COPY src ./src
COPY benchmarks ./benchmarks

RUN pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["codeops-agent", "serve", "--host", "0.0.0.0", "--port", "8000", "--repo", "/app/benchmarks/sample_repo", "--logs", "/app/benchmarks/sample_logs/app.log"]
