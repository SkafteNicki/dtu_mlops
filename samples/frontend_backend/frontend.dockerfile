FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc git && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN mkdir /app

WORKDIR /app

COPY pyproject.toml.frontend /app/pyproject.toml
COPY frontend.py /app/frontend.py

RUN uv pip install --system -r pyproject.toml

EXPOSE $PORT

ENTRYPOINT ["streamlit", "run", "frontend.py", "--server.port", "$PORT", "--server.address=0.0.0.0"]
