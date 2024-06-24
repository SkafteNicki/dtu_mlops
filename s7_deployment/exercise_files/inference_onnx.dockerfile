FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN echo "CUDA is set to: ${CUDA}" && \
    if [ -n "$CUDA" ]; then \
        pip install onnxruntime-gpu; \
    else \
        pip install onnxruntime; \
    fi