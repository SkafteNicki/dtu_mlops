FROM python:3.11-slim

EXPOSE $PORT

WORKDIR /app

COPY requirements.txt .
COPY app.py .

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["sh", "-c", "streamlit run app.py --server.port=$PORT --server.address=0.0.0.0"]
