FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

ARG CUDA
ENV CUDA=${CUDA}
RUN echo "CUDA is set to: ${CUDA}"

RUN echo "CUDA is set to: ${CUDA}" && \
    if [ -n "$CUDA" ]; then \
        pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121; \
    else \
        pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu; \
    fi
