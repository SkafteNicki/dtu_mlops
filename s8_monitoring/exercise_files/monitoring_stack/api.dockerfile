# The API as a container image. Build it from this folder (docker compose up --build does it for you):
#   docker build -f api.dockerfile -t classifier-api .
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.11.27 /uv /bin/uv

WORKDIR /app
# CPU-only torch: about 200 MB instead of several GB with the CUDA libraries.
COPY requirements.txt .
RUN uv pip install --system --no-cache torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu \
    && uv pip install --system --no-cache -r requirements.txt

COPY dataset.py model.py api.py ./
# The model is baked into the image: `uv run python train.py` must have run first.
COPY models/model.pt models/model.pt

# Cloud Run tells the container which port to listen on in $PORT; locally it is 8000.
ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn api:app --host 0.0.0.0 --port $PORT"]
