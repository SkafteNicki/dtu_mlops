from pathlib import Path

import anyio
import nltk
import pandas as pd
from evidently.metric_preset import TargetDriftPreset, TextEvals
from evidently.report import Report
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

nltk.download("words")
nltk.download("wordnet")
nltk.download("omw-1.4")


def run_analysis(reference_data: pd.DataFrame, current_data: pd.DataFrame):
    """Run the analysis and return the report."""
    text_overview_report = Report(metrics=[TextEvals(column_name="content"), TargetDriftPreset(columns=["sentiment"])])
    text_overview_report.run(reference_data=reference_data, current_data=current_data)
    text_overview_report.save("text_overview_report.html")


def lifespan(app: FastAPI):
    """Load the data and class names before the application starts."""
    global training_data, class_names
    training_data = pd.read_csv("reviews.csv")

    def to_sentiment(rating):
        """Convert rating to sentiment class."""
        rating = int(rating)
        if rating <= 2:
            return 0  # Negative
        if rating == 3:
            return 1  # Neutral
        return 2  # Positive

    training_data["sentiment"] = training_data.score.apply(to_sentiment)
    class_names = ["negative", "neutral", "positive"]

    yield

    del training_data, class_names


app = FastAPI(lifespan=lifespan)


def download_files(n: int = 5) -> None:
    """Download the N latest prediction files from the GCP bucket."""


def load_latest_files(directory: Path, n: int) -> pd.DataFrame:
    """Fetch latest data from the database."""
    download_files(n=n)


@app.get("/report")
async def get_report(n: int = 5):
    """Generate and return the report."""
    prediction_data = load_latest_files(Path("."), n=n)
    run_analysis(training_data, prediction_data)

    async with await anyio.open_file("monitoring.html", encoding="utf-8") as f:
        html_content = f.read()

    return HTMLResponse(content=html_content, status_code=200)
