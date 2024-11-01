# Import libraries
import logging
import time

import torch
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from PIL import Image
from prometheus_client import Counter, Summary
from torchvision import models, transforms

# Logging setup for Grafana Loki
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ml-service")

# Initialize FastAPI
app = FastAPI()

# Initialize OpenTelemetry Tracing
resource = Resource.create({"service.name": "image-classifier"})
trace.set_tracer_provider(TracerProvider(resource=resource))
tracer_provider = trace.get_tracer_provider()
otlp_exporter = OTLPSpanExporter(endpoint="http://tempo:4317", insecure=True)
span_processor = BatchSpanProcessor(otlp_exporter)
tracer_provider.add_span_processor(span_processor)

FastAPIInstrumentor.instrument_app(app)
RequestsInstrumentor().instrument()

# Prometheus Metrics
REQUEST_COUNT = Counter("request_count", "App Request Count", ["app_name", "endpoint"])
REQUEST_TIME = Summary("request_processing_seconds", "Time spent processing request")

# Load pre-trained ResNet model
model = models.resnet50(pretrained=True)
model.eval()

# Define ImageNet labels (for simplicity, only showing a few)
imagenet_labels = {0: "tench", 1: "goldfish", 2: "great white shark", 3: "tiger shark", 4: "hammerhead"}

# Define preprocessing for input images
preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    """FastAPI middleware to track request count and duration."""
    start_time = time.time()

    # Request counting for Prometheus
    REQUEST_COUNT.labels(app_name="image-classifier", endpoint=request.url.path).inc()

    # Execute request
    response = await call_next(request)

    # Track the time taken to process the request
    duration = time.time() - start_time
    REQUEST_TIME.observe(duration)

    logger.info(f"Request to {request.url.path} took {duration:.4f}s")
    return response


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict the class of an image."""
    logger.info("Received image for prediction")

    # Ensure the uploaded file is an image
    if file.content_type not in ["image/jpeg", "image/png"]:
        logger.error("Unsupported file type")
        raise HTTPException(status_code=400, detail="Unsupported file type")

    # Read the image
    try:
        image = Image.open(file.file).convert("RGB")
    except Exception as e:
        logger.error(f"Error opening image: {e}")
        raise HTTPException(status_code=500, detail="Error processing image") from e

    # Preprocess the image for the model
    input_tensor = preprocess(image).unsqueeze(0)

    # Perform prediction
    with torch.no_grad():
        output = model(input_tensor)

    # Convert output to probabilities and find the top prediction
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top_prediction = probabilities.argmax().item()

    # Log the prediction result
    logger.info(f"Top prediction: {imagenet_labels.get(top_prediction, 'unknown')}")

    # Tracing this operation
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("image-prediction"):
        return {
            "prediction": imagenet_labels.get(top_prediction, "unknown"),
            "probability": probabilities[top_prediction].item(),
        }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    logger.info("Health check requested")
    return {"status": "healthy"}
