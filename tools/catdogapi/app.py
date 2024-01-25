import os
import random
from io import BytesIO

import fastapi
import requests
from fastapi.responses import FileResponse
from PIL import Image

app = fastapi.FastAPI()


def delete_old_images():
    """Clean up old images."""
    for file in os.listdir():
        if file.startswith("image_"):
            os.remove(file)


@app.get("/")
def index():
    """Home page."""
    return {"message": "Go to /catordog to get a cat or a dog."}


@app.get("/catordog")
def get_cat_or_doc(background_tasks: fastapi.BackgroundTasks):
    """Get a cat or a dog image."""
    background_tasks.add_task(delete_old_images)
    if random.random() > 0.5:
        request = requests.get("https://cataas.com/cat", timeout=10)
        content = request.content
    else:
        request1 = requests.get("https://dog.ceo/api/breeds/image/random", timeout=10)
        path = request1.json()["message"]
        request2 = requests.get(path, timeout=10)
        content = request2.content
    i = Image.open(BytesIO(content))
    h = int(f"{hash(content)}".strip("-"))
    if i.mode in ("RGBA", "P"):
        i = i.convert("RGB")
    i.save(f"image_{h}.jpg")
    return FileResponse(f"image_{h}.jpg")
