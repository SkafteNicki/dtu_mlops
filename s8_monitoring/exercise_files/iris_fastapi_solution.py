import pickle
from collections.abc import Generator
from typing import TYPE_CHECKING

from fastapi import FastAPI

if TYPE_CHECKING:
    from sklearn.neighbors import KNeighborsClassifier


def lifespan(app: FastAPI) -> Generator[None]:
    """Load model and classes."""
    global model, classes
    classes = ["Iris-Setosa", "Iris-Versicolour", "Iris-Virginica"]
    with open("model.pkl", "rb") as file:
        model: KNeighborsClassifier = pickle.load(file)

    yield

    del model, classes


app = FastAPI(lifespan=lifespan)


@app.post("/predict")
def iris_inference(sepal_length: float, sepal_width: float, petal_length: float, petal_width: float):
    """Version 1 of the iris inference endpoint."""
    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = prediction.item()
    return {"prediction": classes[prediction], "prediction_int": prediction}
