FROM python:3.11-slim

WORKDIR /app

RUN pip install fastapi uvicorn nltk evidently "pandas<3" google-cloud-storage --no-cache-dir

COPY sentiment_monitoring.py .

EXPOSE $PORT

CMD exec uvicorn sentiment_monitoring:app --port $PORT --host 0.0.0.0 --workers 1
