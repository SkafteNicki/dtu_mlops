FROM ghcr.io/astral-sh/uv:python3.12-trixie-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc git && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN mkdir /app

WORKDIR /app

COPY pyproject.toml /app/pyproject.toml
COPY backend.py /app/backend.py
COPY imagenet-simple-labels.json /app/imagenet-simple-labels.json

RUN uv sync --group backend --no-install-project --no-dev

EXPOSE 8080
CMD exec uv run --no-sync uvicorn --port $PORT --host 0.0.0.0 backend:app
