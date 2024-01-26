FROM python:3.10-slim

COPY requirements_backend.txt requirements_backend.txt
COPY backend.py backend.py

WORKDIR /

EXPOSE $PORT

RUN pip install -r requirements_backend.txt

CMD exec unicorn --port $PORT --host 0.0.0.0 backend:app
