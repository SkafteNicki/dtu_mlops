import json

import pandas as pd
import scipy
import seaborn as sns
import typer
from matplotlib.pyplot import show, subplots

app = typer.Typer()


@app.command()
def usage_statistics(name: str = "responses.json") -> None:
    """Plot histograms of the request and response tokens."""
    with open(name) as file:
        data = json.load(file)

    request_tokens: list[float] = []
    response_tokens: list[float] = []
    for d in data:
        request_tokens.append(d["request_usage"]["request_tokens"])
        response_tokens.append(d["request_usage"]["response_tokens"])

    dataframe = pd.DataFrame({"request_tokens": request_tokens, "response_tokens": response_tokens})
    sns.displot(dataframe, x="request_tokens")
    sns.displot(dataframe, x="response_tokens")

    r, p = scipy.stats.pearsonr(request_tokens, response_tokens)
    plot = sns.lmplot(data=dataframe, x="request_tokens", y="response_tokens")
    plot.figure.suptitle(f"Pearson correlation: {r:.2f}, p-value: {p:.2f}")

    show()


@app.command()
def score_statistics(name: str = "responses.json") -> None:
    """Plot histograms of the scores."""
    with open(name) as file:
        data = json.load(file)

    code_quality: list[float] = []
    unit_testing: list[float] = []
    ci_cd: list[float] = []
    overall_score: list[float] = []
    confidence: list[float] = []
    for d in data:
        code_quality.append(d["code_quality"])
        unit_testing.append(d["unit_testing"])
        ci_cd.append(d["ci_cd"])
        overall_score.append(d["overall_score"])
        confidence.append(d["confidence"])

    dataframe = pd.DataFrame(
        {
            "code_quality": code_quality,
            "unit_testing": unit_testing,
            "ci_cd": ci_cd,
            "overall_score": overall_score,
            "confidence": confidence,
        }
    )
    _, axes = subplots(1, 5)
    sns.histplot(dataframe, x="code_quality", ax=axes[0], bins=[1, 2, 3, 4, 5])
    sns.histplot(dataframe, x="unit_testing", ax=axes[1], bins=[1, 2, 3, 4, 5])
    sns.histplot(dataframe, x="ci_cd", ax=axes[2], bins=[1, 2, 3, 4, 5])
    sns.histplot(dataframe, x="overall_score", ax=axes[3], bins=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    sns.histplot(dataframe, x="confidence", ax=axes[4], bins=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    show()


if __name__ == "__main__":
    app()
