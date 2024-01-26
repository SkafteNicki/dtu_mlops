FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY app.py app.py

EXPOSE $PORT
CMD exec uvicorn app:app --host 0.0.0.0 --port $PORT
