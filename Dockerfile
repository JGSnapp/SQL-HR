# syntax=docker/dockerfile:1

FROM python:3.11-slim

ENV VLLM_TARGET_DEVICE=cpu
ENV VLLM_CPU_OMP_THREADS_BIND=nobind

RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "vllm[openai]" && \
    python - <<'PY'
import importlib.metadata as md
import pathlib

dist = md.distribution("vllm")
metadata_path = pathlib.Path(dist._path) / "METADATA"  # type: ignore[attr-defined]
version = dist.version
if "+cpu" not in version and metadata_path.exists():
    text = metadata_path.read_text()
    if f"Version: {version}" in text:
        metadata_path.write_text(text.replace(f"Version: {version}", f"Version: {version}+cpu", 1))
PY

WORKDIR /app
RUN mkdir -p /models

ENTRYPOINT ["python", "-m", "vllm.entrypoints.openai.api_server"]

CMD ["--model", "t-tech/T-lite-it-1.0", "--host", "0.0.0.0", "--port", "8000", "--served-model-name", "local-llama", "--api-key", "local-key-123", "--dtype", "float32", "--max-model-len", "2048", "--download-dir", "/models", "--enforce-eager"]
