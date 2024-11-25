import base64
import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google.cloud.storage import Client
from models import RepoStats
from PIL import Image

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


def activity_to_image(activity_matrix: list[list[int]], scale_factor: int = 10) -> str:
    """
    Convert an activity matrix (N, 24) into an RGB image scaled up by a given factor.

    Args:
        activity_matrix (list[list[int]]): A 2D list of activity values.
        scale_factor (int): Factor by which to scale up the image size.

    Returns:
        str: Base64-encoded PNG image string in "data:image/png;base64," format.
    """
    # Normalize the activity matrix to the range [0, 255].
    array = np.array(activity_matrix, dtype=np.float32)
    max_value = np.max(array)
    if max_value > 0:
        array = array / max_value * 255

    # Create an RGB image: Green for activity, Black for no activity.
    height, width = array.shape
    rgb_array = np.zeros((height, width, 3), dtype=np.uint8)
    rgb_array[:, :, 1] = array.astype(np.uint8)  # Green channel

    # Scale up the image by the scale factor.
    scaled_height, scaled_width = height * scale_factor, width * scale_factor
    image = Image.fromarray(rgb_array, mode="RGB")
    image = image.resize((scaled_width, scaled_height), Image.NEAREST)

    # Convert the image to a Base64 string.
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode("utf-8")

    return f"data:image/png;base64,{image_base64}"


def main() -> None:
    """Main function for the leaderboard."""
    download_data("repo_stats.json")
    dataframe = load_data("repo_stats.json")
    dataframe["num_warnings"] = dataframe["num_warnings"].apply(lambda x: 27 - x if pd.notnull(x) else x)
    dataframe["activity_matrix"] = dataframe["activity_matrix"].apply(
        lambda x: activity_to_image(x) if x is not None else x
    )
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
            "activity_matrix",
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
            "actions_passing",
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
            "num_contributors": "Contributors",
            "total_commits": "Total Commits",
            "num_commits_to_main": "Commits to main",
            "contributions_per_contributor": st.column_config.BarChartColumn("Contributions distribution"),
            "num_prs": "PRs",
            "average_commit_length_to_main": "ACML* (main)",
            "average_commit_length": "ACML* (all)",
            "latest_commit": st.column_config.DatetimeColumn("Latest commit"),
            "activity_matrix": st.column_config.ImageColumn(
                "Commit activity**",
                width="medium",
            ),
        },
        hide_index=True,
    )
    st.write("*ACML = Average commit message length")
    st.write(
        "**Activity matrix is a (N, 24) matrix where N is the number of days since the first commit."
        " Each row represents the number of commits per hour for that day."
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
            "actions_passing": "Actions passing",
            "readme_length": "Readme size",
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
