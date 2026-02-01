FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

EXPOSE $PORT

WORKDIR /app

COPY pyproject.toml .
COPY leaderboard.py .
COPY models.py .

RUN uv pip install --system -r pyproject.toml

# Command to run the Streamlit application
ENTRYPOINT ["sh", "-c", "streamlit run leaderboard.py --server.port=$PORT --server.address=0.0.0.0"]
