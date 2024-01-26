import json

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import models, transforms

app = FastAPI()

# Load the pre-trained ResNet model
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model.eval()

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

imagenet_classes = json.load(open("imagenet-simple-labels.json"))


def predict_image(image_path: str) -> str:
    """Predict image class (or classes) given image path and return the result."""
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(img)
    _, predicted_idx = torch.max(output, 1)
    return imagenet_classes[predicted_idx.item()]


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
        with open(file.filename, "wb") as f:
            f.write(contents)
        prediction = predict_image(file.filename)
        return {"filename": file.filename, "prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500) from e
