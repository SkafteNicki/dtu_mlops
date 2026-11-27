FROM ghcr.io/astral-sh/uv:python3.12-trixie-slim

EXPOSE 8080

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/streamlit/streamlit-example.git .

RUN uv pip install --system -r requirements.txt

CMD exec streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
