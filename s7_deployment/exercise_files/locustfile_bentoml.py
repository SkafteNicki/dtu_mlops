import numpy as np
from locust import HttpUser, between, task
from PIL import Image


def prepare_image():
    """Load and preprocess the image as required."""
    image = Image.open("my_cat.jpg")
    image = image.resize((224, 224))
    image = np.array(image)
    image = np.transpose(image, (2, 0, 1))  # Convert to CHW format
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    # Convert to list format for JSON serialization
    return image.tolist()


image = prepare_image()


class BentoMLUser(HttpUser):
    """Locust user class for sending prediction requests to the server."""

    wait_time = between(1, 2)

    @task
    def send_prediction_request(self):
        """Send a prediction request to the server."""
        payload = {"image": image}  # Package the image as JSON
        self.client.post("/predict", json=payload, headers={"Content-Type": "application/json"})
