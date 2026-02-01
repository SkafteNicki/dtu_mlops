FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc git && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN mkdir /app

WORKDIR /app

COPY pyproject.toml.backend /app/pyproject.toml
COPY backend.py /app/backend.py

RUN uv pip install --system -r pyproject.toml

EXPOSE $PORT
CMD exec unicorn --port $PORT --host 0.0.0.0 backend:app
