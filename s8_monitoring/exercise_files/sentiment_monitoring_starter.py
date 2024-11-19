import nltk
import pandas as pd
from evidently.presents import TextOverviewPreset
from evidently.report import Report
from fastapi import FastAPI

nltk.download("words")
nltk.download("wordnet")
nltk.download("omw-1.4")


def run_analysis(reference_data: pd.DataFrame, current_data: pd.DataFrame):
    """Run the analysis and return the report."""
    text_overview_report = Report(metrics=[TextOverviewPreset(column_name="Review_Text")])
    text_overview_report.run(reference_data=reference_data, current_data=current_data)


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


def fetch_latest_data():
    """Fetch latest data from the database."""


@app.get("/report")
async def get_report():
    """Generate and return the report."""
    prediction_data = fetch_latest_data()
    return run_analysis(training_data, prediction_data)
