FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
RUN uv pip install --system --no-cache fastapi uvicorn scikit-learn

COPY model_versions_app.py model_versions_app.py

# Cloud Run sets $PORT, default to 8080 locally. `exec` makes uvicorn PID 1 so it receives SIGTERM.
CMD exec uvicorn model_versions_app:app --host 0.0.0.0 --port ${PORT:-8080}
