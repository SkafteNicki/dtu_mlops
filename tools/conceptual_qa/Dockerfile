FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

EXPOSE $PORT

WORKDIR /app

COPY pyproject.toml .
COPY app.py .

RUN uv pip install --system -r pyproject.toml

ENTRYPOINT ["sh", "-c", "streamlit run app.py --server.port=$PORT --server.address=0.0.0.0"]
