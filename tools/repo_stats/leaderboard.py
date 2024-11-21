import json
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google.cloud.storage import Client
from models import RepoStats

load_dotenv()


def download_data(file_name: str) -> None:
    """Downloads the group-repository data from GCS."""
    storage_client = Client()
    bucket = storage_client.bucket("mlops_group_repository")
    blob = bucket.blob(file_name)
    blob.download_to_filename(file_name)


def load_data(file_name: str) -> pd.DataFrame:
    """Loads the group-repository data into a DataFrame."""
    with Path(file_name).open() as f:
        content = json.load(f)
    repo_content = [RepoStats(**group) for group in content]
    repo_content_dicts = [repo.model_dump() for repo in repo_content]
    return pd.DataFrame(repo_content_dicts)


def main() -> None:
    """Main function for the leaderboard."""
    download_data("repo_stats.json")
    dataframe = load_data("repo_stats.json")
    dataframe["num_warnings"] = dataframe["num_warnings"].apply(lambda x: 27 - x if pd.notnull(x) else x)

    st.set_page_config(layout="wide")
    st.title("Group Github Stats")
    st.text(
        """
        Below is shown automatic scraped data for all groups in the course. None of these stats directly contribute
        towards you passing the course or not. Instead they can inform how you are doing in comparison to other groups,
        and it can indirectly inform the us about how well you are using version control for collaborating on your
        project.
        """,
    )

    df_base = dataframe[
        [
            "group_number",
            "group_size",
            "num_contributors",
            "total_commits",
            "num_commits_to_main",
            "contributions_per_contributor",
            "num_prs",
            "average_commit_length_to_main",
            "average_commit_length",
            "latest_commit",
        ]
    ]

    df_content = dataframe[
        [
            "group_number",
            "num_python_files",
            "num_docker_files",
            "num_workflow_files",
            "has_requirements_file",
            "has_cloudbuild",
            "using_dvc",
            "repo_size",
            "readme_length",
            "num_warnings",
        ]
    ]

    st.header("Base statistics")
    st.dataframe(
        df_base,
        column_config={
            "group_number": "Group Number",
            "group_size": "Group Size",
            "num_contributors": "Number of contributors",
            "total_commits": "Total Commits",
            "num_commits_to_main": "Commits to main",
            "contributions_per_contributor": st.column_config.BarChartColumn("Contributions distribution"),
            "num_prs": "Number of Pull Requests",
            "average_commit_length_to_main": "ACML* (main)",
            "average_commit_length": "ACML* (all)",
            "latest_commit": st.column_config.DatetimeColumn("Latest commit"),
        },
        hide_index=True,
    )

    st.header("Content statistics")
    st.dataframe(
        df_content,
        column_config={
            "group_number": "Group Number",
            "num_python_files": "Python files",
            "num_docker_files": "Docker files",
            "num_workflow_files": "Workflow files",
            "has_requirements_file": "Requirement file",
            "has_cloudbuild": "Cloudbuild",
            "using_dvc": "Using dvc",
            "repo_size": "Repository size",
            "readme_size": "Readme size",
            "num_warnings": st.column_config.ProgressColumn(
                "Report completion",
                help="Number of questions answered in exam report",
                format="%d",
                min_value=0,
                max_value=27,
            ),
        },
        hide_index=True,
    )


if __name__ == "__main__":
    main()
