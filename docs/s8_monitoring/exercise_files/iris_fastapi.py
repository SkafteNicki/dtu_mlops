from fastapi import FastAPI
import pickle
from datetime import datetime

app = FastAPI()

classes =  ["Iris-Setosa", "Iris-Versicolour", "Iris-Virginica"]
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.post("/iris_v1/")
def iris_inference_v1(
    sepal_length: float,
    sepal_width: float,
    petal_length: float,
    petal_width: float
):
    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = prediction.item()
    return {"prediction": classes[prediction], "prediction_int": prediction}

with open('prediction_database.csv', 'w') as file:
    file.write("time, sepal_length, sepal_width, petal_length, petal_width, prediction\n")

def add_to_database(
    now: str, sepal_length: float, sepal_width: float, petal_length: float, petal_width: float, prediction: int):
    with open('prediction_database.csv', 'a') as file:
        file.write(f"{now}, {sepal_length}, {sepal_width}, {petal_length}, {petal_width}, {prediction}\n")

from fastapi import BackgroundTasks

@app.post("/iris_v2/")
async def iris_inference_v2(
    sepal_length: float,
    sepal_width: float,
    petal_length: float,
    petal_width: float,
    background_tasks: BackgroundTasks,
):
    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = prediction.item()

    now = str(datetime.now())
    background_tasks.add_task(add_to_database, now, sepal_length, sepal_width, petal_length, petal_width, prediction)

    return {"prediction": classes[prediction], "prediction_int": prediction}

from fastapi.responses import HTMLResponse

from sklearn import datasets

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset


@app.get("/iris_monitoring/", response_class=HTMLResponse)
async def iris_monitoring(

):
    iris_frame = datasets.load_iris(as_frame='auto').frame

    data_drift_report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset(),
        TargetDriftPreset(),
    ])

    data_drift_report.run(current_data=iris_frame.iloc[:60], reference_data=iris_frame.iloc[60:], column_mapping=None)
    data_drift_report.save_html('monitoring.html')

    with open("monitoring.html", "r", encoding='utf-8') as f:
        html_content = f.read()

    return HTMLResponse(content=html_content, status_code=200)
