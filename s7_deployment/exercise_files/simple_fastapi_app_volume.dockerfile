FROM python:3.11-slim

EXPOSE $PORT

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install fastapi
RUN pip install pydantic
RUN pip install uvicorn
RUN pip install python-multipart

COPY simple_fastapi_app_volume.py simple_fastapi_app_volume.py

CMD exec uvicorn simple_fastapi_app_volume:app --port $PORT --host 0.0.0.0 --workers 1
