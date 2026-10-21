import json
from contextlib import asynccontextmanager

import anyio
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import models, transforms


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Context manager to start and stop the lifespan events of the FastAPI application."""
    global model, transform, imagenet_classes
    # Load model
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ],
    )

    async with await anyio.open_file("imagenet-simple-labels.json") as f:
        imagenet_classes = json.load(f)

    yield

    # Clean up
    del model
    del transform
    del imagenet_classes


app = FastAPI(lifespan=lifespan)


def predict_image(image_path: str) -> str:
    """Predict image class (or classes) given image path and return the result."""
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(img)
    _, predicted_idx = torch.max(output, 1)
    return output.softmax(dim=-1), imagenet_classes[predicted_idx.item()]


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Hello from the backend!"}


# FastAPI endpoint for image classification
@app.post("/classify/")
async def classify_image(file: UploadFile = File(...)):
    """Classify image endpoint."""
    try:
        contents = await file.read()
        async with await anyio.open_file(file.filename, "wb") as f:
            f.write(contents)
        probabilities, prediction = predict_image(file.filename)
        return {"filename": file.filename, "prediction": prediction, "probabilities": probabilities.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500) from e
