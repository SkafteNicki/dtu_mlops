import bentoml
import numpy as np
from PIL import Image

if __name__ == "__main__":
    image = Image.open("my_cat.jpg")
    image = image.resize((224, 224))  # Resize to match the minimum input size of the model
    image = np.array(image)
    image = np.transpose(image, (2, 0, 1))  # Change to CHW format
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    with bentoml.SyncHTTPClient("http://localhost:4040") as client:
        resp = client.predict(image=image)
        print(resp)
