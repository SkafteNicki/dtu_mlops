FROM python:3.11-slim

WORKDIR /app

RUN pip install fastapi torch transformers google-cloud-storage pydantic --no-cache-dir

COPY sentiment_api.py .

EXPOSE $PORT

CMD exec uvicorn sentiment_api:app --port $PORT --host 0.0.0.0 --workers 1
