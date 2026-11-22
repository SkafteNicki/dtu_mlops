from pathlib import Path

import anyio
import nltk
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.descriptors import OOVWordsPercentage, Sentiment, TextLength
from evidently.metrics import ValueDrift
from evidently.presets import DataDriftPreset, TextEvals
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

nltk.download("words")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("vader_lexicon")


def run_analysis(reference_data: pd.DataFrame, current_data: pd.DataFrame) -> None:
    """Run the analysis and save the report."""
    data_definition = DataDefinition(text_columns=["content"], categorical_columns=["sentiment"])
    # Descriptors turn the raw text into numerical features (one per row) that we can monitor
    descriptors = [
        TextLength("content", alias="Length"),
        OOVWordsPercentage("content", alias="OOV words %"),
        Sentiment("content", alias="Text sentiment"),
    ]
    reference = Dataset.from_pandas(
        reference_data[["content", "sentiment"]], data_definition=data_definition, descriptors=descriptors
    )
    current = Dataset.from_pandas(
        current_data[["content", "sentiment"]], data_definition=data_definition, descriptors=descriptors
    )

    report = Report(
        metrics=[
            TextEvals(),  # summary of the text descriptors
            DataDriftPreset(columns=["Length", "OOV words %", "Text sentiment"]),  # drift of the text descriptors
            ValueDrift(column="sentiment"),  # drift of the predicted sentiment (target drift)
        ]
    )
    snapshot = report.run(current_data=current, reference_data=reference)
    snapshot.save_html("monitoring.html")


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
        html_content = await f.read()

    return HTMLResponse(content=html_content, status_code=200)
