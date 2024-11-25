FROM python:3.11-slim

EXPOSE $PORT

WORKDIR /app

COPY requirements.txt .
COPY leaderboard.py .
COPY models.py .

RUN pip install --no-cache-dir -r requirements.txt

# Command to run the Streamlit application
ENTRYPOINT ["sh", "-c", "streamlit run leaderboard.py --server.port=$PORT --server.address=0.0.0.0"]
