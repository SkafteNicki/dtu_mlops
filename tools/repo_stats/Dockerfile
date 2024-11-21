FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
COPY leaderboard.py .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE $PORT

# Command to run the Streamlit application
CMD ["streamlit", "run", "--server.port", "$PORT", "leaderboard.py"]
