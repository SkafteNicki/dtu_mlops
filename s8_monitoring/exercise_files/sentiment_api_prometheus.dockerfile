FROM python:3.11-slim

WORKDIR /app

RUN pip install fastapi uvicorn torch transformers google-cloud-storage pydantic prometheus-client --no-cache-dir

COPY sentiment_api_prometheus_advance.py .

EXPOSE $PORT

CMD exec uvicorn sentiment_api_prometheus_advance:app --port $PORT --host 0.0.0.0 --workers 1
