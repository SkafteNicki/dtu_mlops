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
    file.write("time, sepal_length, sepal_width, petal_length, petal_width\n")

def add_to_database(now: str, sepal_length: float, sepal_width: float, petal_length: float, petal_width: float):
    with open('prediction_database.csv', 'a') as file:
        file.write(f"{now}, {sepal_length}, {sepal_width}, {petal_length}, {petal_width}\n")

from fastapi import BackgroundTasks

@app.post("/iris_v2/")
async def iris_inference_v2(
    sepal_length: float,
    sepal_width: float,
    petal_length: float,
    petal_width: float,
    background_tasks: BackgroundTasks,
):
    now = str(datetime.now())
    background_tasks.add_task(add_to_database, now, sepal_length, sepal_width, petal_length, petal_width)
    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = prediction.item()
    return {"prediction": classes[prediction], "prediction_int": prediction}
