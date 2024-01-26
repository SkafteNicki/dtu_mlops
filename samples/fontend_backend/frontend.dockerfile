# Use an official Python runtime as a parent image
FROM python:3.10-slim

WORKDIR /app

COPY requirements_frontend.txt requirements_frontend.txt
COPY fontend.py fontend.py

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE $PORT

# Run streamlit when the container launches
CMD ["streamlit", "run", "your_streamlit_app.py"]
