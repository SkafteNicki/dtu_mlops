import json
import os
from pathlib import Path

import nltk
import pandas as pd
from evidently.metric_preset import TargetDriftPreset, TextEvals
from evidently.report import Report
from fastapi import FastAPI

nltk.download("words")
nltk.download("wordnet")
nltk.download("omw-1.4")


def to_sentiment(rating):
    """Convert rating to sentiment class."""
    rating = int(rating)
    if rating <= 2:
        return 0  # Negative
    if rating == 3:
        return 1  # Neutral
    return 2  # Positive


def sentiment_to_numeric(sentiment: str) -> int:
    """Convert sentiment class to numeric."""
    if sentiment == "negative":
        return 0
    if sentiment == "neutral":
        return 1
    return 2


def run_analysis(reference_data: pd.DataFrame, current_data: pd.DataFrame):
    """Run the analysis and return the report."""
    text_overview_report = Report(metrics=[TextEvals(column_name="content"), TargetDriftPreset(columns=["sentiment"])])
    text_overview_report.run(reference_data=reference_data, current_data=current_data)
    text_overview_report.save("text_overview_report.html")


def lifespan(app: FastAPI):
    """Load the data and class names before the application starts."""
    global training_data, class_names
    training_data = pd.read_csv("reviews.csv")
    training_data["sentiment"] = training_data.score.apply(to_sentiment)
    training_data["target"] = training_data["sentiment"]  # evidently expects the target column to be named "target"
    class_names = ["negative", "neutral", "positive"]

    yield

    del training_data, class_names


app = FastAPI(lifespan=lifespan)


def load_latest_files(directory: Path, n: int) -> pd.DataFrame:
    """Load the N latest prediction files from the directory."""
    # Get all prediction files in the directory
    files = directory.glob("prediction_*.json")

    # Sort files based on when they where created
    files = sorted(files, key=os.path.getmtime)

    # Get the N latest files
    latest_files = files[-n:]

    # Load or process the files as needed
    reviews, sentiment = [], []
    for file in latest_files:
        with file.open() as f:
            data = json.load(f)
            reviews.append(data["review"])
            sentiment.append(sentiment_to_numeric(data["sentiment"]))
    dataframe = pd.DataFrame({"content": reviews, "sentiment": sentiment})
    dataframe["target"] = dataframe["sentiment"]
    return dataframe


@app.get("/")
@app.get("/report")
async def get_report(n: int = 5):
    """Generate and return the report."""
    prediction_data = load_latest_files(Path("."), n=n)
    return run_analysis(training_data, prediction_data)
