FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc git && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN mkdir /app

WORKDIR /app

COPY pyproject.toml /app/pyproject.toml
COPY frontend.py /app/frontend.py

RUN uv sync --group frontend --no-install-project --no-dev

EXPOSE $PORT

ENTRYPOINT ["uv", "run", "streamlit", "run", "frontend.py", "--server.port", "$PORT", "--server.address=0.0.0.0"]
