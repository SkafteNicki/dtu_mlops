from io import BytesIO

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision import transforms
from yolov5 import Model

app = FastAPI()

# Load the YOLOv5 model
model = Model(weights="yolov5s.pt")


def predict_object(image: torch.Tensor) -> dict:
    # Perform inference using the YOLOv5 model
    results = model(image)

    # Process the results as needed
    # For example, you might want to filter out low-confidence predictions
    # or convert the results to a specific format

    # For simplicity, we'll return the raw results in this example
    return results


def process_image(file: UploadFile) -> torch.Tensor:
    # Open and preprocess the uploaded image
    image = Image.open(BytesIO(file.file.read())).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize((416, 416)),
            transforms.ToTensor(),
        ]
    )
    image = transform(image).unsqueeze(0)

    return image


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Process the uploaded image
        image = process_image(file)

        # Make predictions using the YOLOv5 model
        predictions = predict_object(image)

        # Convert predictions to JSON and return
        predictions_json = jsonable_encoder(predictions)
        return JSONResponse(content=predictions_json)

    except Exception as e:
        # Handle errors and return appropriate HTTP response
        return HTTPException(status_code=500, detail=str(e))
