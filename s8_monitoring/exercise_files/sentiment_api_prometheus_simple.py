import datetime
import json
import os
from contextlib import asynccontextmanager

import torch
import torch.nn as nn
from fastapi import BackgroundTasks, FastAPI, HTTPException
from google.cloud import storage
from prometheus_client import Counter, make_asgi_app
from pydantic import BaseModel
from transformers import BertModel, BertTokenizer

# Define model and device configuration
BUCKET_NAME = "gcp_monitoring_exercise"
MODEL_NAME = "bert-base-cased"
MODEL_FILE_NAME = "bert_sentiment_model.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReviewInput(BaseModel):
    """Define input data structure for the endpoint."""

    review: str


class PredictionOutput(BaseModel):
    """Define output data structure for the endpoint."""

    sentiment: str


class SentimentClassifier(nn.Module):
    """Sentiment Classifier class. Combines BERT model with a dropout and linear layer."""

    def __init__(self, n_classes, model_name=MODEL_NAME):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        """Forward pass of the model."""
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        output = self.drop(output[1])
        return self.out(output)


# Define Prometheus metrics
error_counter = Counter("prediction_error", "Number of prediction errors")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model and tokenizer when the app starts and clean up when the app stops."""
    global model, tokenizer, class_names
    if "bert_sentiment_model.pt" not in os.listdir():
        download_model_from_gcp()  # Download the model from GCP
    model = SentimentClassifier(n_classes=3)
    model.load_state_dict(torch.load("bert_sentiment_model.pt", map_location=device))
    model = model.to(device)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    class_names = ["negative", "neutral", "positive"]
    print("Model and tokenizer loaded successfully")

    yield

    del model, tokenizer


# Initialize FastAPI app
app = FastAPI(lifespan=lifespan)
app.mount("/metrics", make_asgi_app())


def download_model_from_gcp():
    """Download the model from GCP bucket."""
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(MODEL_FILE_NAME)
    blob.download_to_filename(MODEL_FILE_NAME)
    print(f"Model {MODEL_FILE_NAME} downloaded from GCP bucket {BUCKET_NAME}.")


# Save prediction results to GCP
def save_prediction_to_gcp(review: str, outputs: list[float], sentiment: str):
    """Save the prediction results to GCP bucket."""
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    time = datetime.datetime.now(tz=datetime.UTC)
    # Prepare prediction data
    data = {
        "review": review,
        "sentiment": sentiment,
        "probability": outputs,
        "timestamp": datetime.datetime.now(tz=datetime.UTC).isoformat(),
    }
    blob = bucket.blob(f"prediction_{time}.json")
    blob.upload_from_string(json.dumps(data))
    print("Prediction saved to GCP bucket.")


# Prediction endpoint
@app.post("/predict", response_model=PredictionOutput)
async def predict_sentiment(review_input: ReviewInput, background_tasks: BackgroundTasks):
    """Predict sentiment of the input text."""
    try:
        # Encode input text
        encoding = tokenizer.encode_plus(
            review_input.review,
            add_special_tokens=True,
            max_length=160,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        # Model prediction
        with torch.no_grad():
            outputs: torch.Tensor = model(input_ids, attention_mask)
            _, prediction = torch.max(outputs, dim=1)
            sentiment = class_names[prediction]

        background_tasks.add_task(save_prediction_to_gcp, review_input.review, outputs.softmax(-1).tolist(), sentiment)

        return PredictionOutput(sentiment=sentiment)

    except Exception as e:
        error_counter.inc()
        raise HTTPException(status_code=500, detail=str(e)) from e
