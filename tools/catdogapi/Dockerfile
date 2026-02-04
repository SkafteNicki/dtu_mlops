FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

COPY pyproject.toml pyproject.toml
RUN uv pip install --system -r pyproject.toml

COPY app.py app.py

EXPOSE $PORT
CMD exec uvicorn app:app --host 0.0.0.0 --port $PORT
