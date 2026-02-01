FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md
COPY main.py main.py
WORKDIR /
RUN uv sync --locked

ENTRYPOINT ["python", "-u", "main.py"]
