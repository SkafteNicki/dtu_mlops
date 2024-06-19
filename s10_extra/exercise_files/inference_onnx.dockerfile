FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir onnxruntime

COPY inference.py /app/inference.py
COPY model.onnx /app/model.onnx

WORKDIR /app

CMD ["python", "inference.py"]
