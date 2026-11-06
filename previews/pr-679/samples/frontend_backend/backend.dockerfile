FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc git && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN mkdir /app

WORKDIR /app

COPY requirements_backend.txt /app/requirements_backend.txt
COPY backend.py /app/backend.py

RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements_backend.txt

EXPOSE $PORT
CMD exec unicorn --port $PORT --host 0.0.0.0 backend:app
