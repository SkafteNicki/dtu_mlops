import pickle
from collections.abc import Generator
from datetime import datetime
from typing import TYPE_CHECKING

import anyio
import pandas as pd
from evidently.metric_preset import (
    DataDriftPreset,
    DataQualityPreset,
    TargetDriftPreset,
)
from evidently.report import Report
from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import HTMLResponse
from sklearn import datasets

if TYPE_CHECKING:
    from sklearn.neighbors import KNeighborsClassifier


def lifespan(app: FastAPI) -> Generator[None]:
    """Load model and classes, and create database file."""
    global model, classes
    classes = ["Iris-Setosa", "Iris-Versicolour", "Iris-Virginica"]
    with open("model.pkl", "rb") as file:
        model: KNeighborsClassifier = pickle.load(file)

    with open("prediction_database.csv", "w") as file:
        file.write("time, sepal_length, sepal_width, petal_length, petal_width, prediction\n")

    yield

    del model


app = FastAPI(lifespan=lifespan)


def add_to_database(
    now: str,
    sepal_length: float,
    sepal_width: float,
    petal_length: float,
    petal_width: float,
    prediction: int,
) -> None:
    """Simple function to add prediction to database."""
    with open("prediction_database.csv", "a") as file:
        file.write(f"{now}, {sepal_length}, {sepal_width}, {petal_length}, {petal_width}, {prediction}\n")


@app.post("/predict")
async def iris_inference(
    sepal_length: float,
    sepal_width: float,
    petal_length: float,
    petal_width: float,
    background_tasks: BackgroundTasks,
):
    """Version 2 of the iris inference endpoint."""
    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = prediction.item()

    now = str(datetime.now(tz=datetime.UTC))
    background_tasks.add_task(add_to_database, now, sepal_length, sepal_width, petal_length, petal_width, prediction)
    return {"prediction": classes[prediction], "prediction_int": prediction}


@app.get("/monitoring", response_class=HTMLResponse)
async def iris_monitoring():
    """Simple get request method that returns a monitoring report."""
    reference_data: pd.DataFrame = datasets.load_iris(as_frame=True).frame
    reference_data = reference_data.rename(
        columns={
            "sepal length (cm)": "sepal_length",
            "sepal width (cm)": "sepal_width",
            "petal length (cm)": "petal_length",
            "petal width (cm)": "petal_width",
            "target": "target",
        }
    )
    current_data = pd.read_csv("prediction_database.csv")
    current_data = current_data.drop(columns=["time"])

    data_drift_report = Report(metrics=[DataDriftPreset(), DataQualityPreset(), TargetDriftPreset()])
    data_drift_report.run(current_data=current_data, reference_data=reference_data)
    data_drift_report.save_html("monitoring.html")

    async with await anyio.open_file("monitoring.html", encoding="utf-8") as f:
        html_content = f.read()

    return HTMLResponse(content=html_content, status_code=200)
