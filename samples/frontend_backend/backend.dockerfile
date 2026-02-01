FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc git && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN mkdir /app

WORKDIR /app

COPY pyproject.toml /app/pyproject.toml
COPY backend.py /app/backend.py

RUN uv sync --group backend --no-install-project --no-dev

EXPOSE $PORT
CMD exec uvicorn --port $PORT --host 0.0.0.0 backend:app
