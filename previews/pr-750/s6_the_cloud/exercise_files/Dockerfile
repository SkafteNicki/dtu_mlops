FROM ghcr.io/astral-sh/uv:python3.12-trixie-slim

# install build tools
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml pyproject.toml
COPY main.py main.py
RUN uv sync --no-install-project --no-cache

ENTRYPOINT ["uv", "run", "--no-sync", "main.py"]
