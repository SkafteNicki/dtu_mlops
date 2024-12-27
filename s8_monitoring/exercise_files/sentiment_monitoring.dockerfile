FROM python:3.11-slim

WORKDIR /app

RUN pip install fastapi nltk evidently google-cloud-storage --no-cache-dir

COPY sentiment_monitoring.py .

EXPOSE $PORT

CMD exec uvicorn sentiment_monitoring:app --port $PORT --host 0.0.0.0 --workers 1
