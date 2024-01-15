r"""Basic streamlit leaderboard app for showing data from scraped github repos.

Run with:
    streamlit run tools\leaderboard_app\leaderboard_app.py
"""

import ast
import os
import sys
from datetime import datetime

import dropbox
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from dropbox.exceptions import AuthError

st.set_page_config(layout="wide")

if st.secrets.load_if_toml_exists():
    DROPBOX_TOKEN = st.secrets["DROPBOX_TOKEN"]
    DROPBOX_APP_KEY = st.secrets["DROPBOX_APP_KEY"]
    DROPBOX_APP_SECRET = st.secrets["DROPBOX_APP_SECRET"]
    DROPBOX_REFRESH_TOKEN = st.secrets["DROPBOX_REFRESH_TOKEN"]
else:  # load credentials from .env file
    load_dotenv()
    DROPBOX_TOKEN = os.getenv("DROPBOX_TOKEN")
    DROPBOX_APP_KEY = os.getenv("DROPBOX_APP_KEY")
    DROPBOX_APP_SECRET = os.getenv("DROPBOX_APP_SECRET")
    DROPBOX_REFRESH_TOKEN = os.getenv("DROPBOX_REFRESH_TOKEN")


def download_data(filename: str) -> None:
    """Download data from dropbox."""
    with dropbox.Dropbox(
        oauth2_access_token=DROPBOX_TOKEN,
        app_key=DROPBOX_APP_KEY,
        app_secret=DROPBOX_APP_SECRET,
        oauth2_refresh_token=DROPBOX_REFRESH_TOKEN,
    ) as dbx:
        try:
            dbx.users_get_current_account()
        except AuthError:
            sys.exit("ERROR: Invalid access token; try re-generating an access token from the app console on the web.")

        dbx.files_download_to_file(filename, f"/{filename}")


def main():
    """Streamlit application for showing group github stats."""
    download_data("latest_repo_data.csv")

    df = pd.read_csv("latest_repo_data.csv")

    # convert to column
    df["contributions_per_contributor"] = df["contributions_per_contributor"].apply(
        lambda x: ast.literal_eval(x) if pd.notnull(x) else x
    )
    df["warnings_raised"] = df["warnings_raised"].apply(lambda x: 27 - x if pd.notnull(x) else x)
    f = "%Y-%m-%dT%H:%M:%SZ"
    df["latest_commit"] = df["latest_commit"].apply(
        lambda x: datetime.strptime(x, f) if pd.notnull(x) else x,
    )

    # remove columns that are not needed
    df1 = df[
        [
            "group_nb",
            "num_students",
            "num_contributors",
            "total_commits",
            "num_commits_to_main",
            "contributions_per_contributor",
            "num_prs",
            "average_commit_message_length_to_main",
            "average_commit_message_length",
            "latest_commit",
        ]
    ]

    df2 = df[
        [
            "group_nb",
            "num_docker_files",
            "num_workflow_files",
            "has_requirement_file",
            "has_makefile",
            "has_cloudbuild",
            "repo_size",
            "readme_size",
            "using_dvc",
            "warnings_raised",
        ]
    ]

    st.title("Group Github Stats")
    st.text(
        """
        Below is shown automatic scraped data for all groups in the course. None of these stats directly contribute
        towards you passing the course or not. Instead they can inform how you are doing in comparison to other groups,
        and it can indirectly inform the us about how well you are using version control for collaborating on your
        project.
        """
    )

    st.header("Base statistics")
    st.dataframe(
        df1,
        column_config={
            "group_nb": "Group Number",
            "num_students": "Students",
            "num_contributors": "Contributors",
            "total_commits": "Total Commits",
            "num_commits_to_main": "Commits to main",
            "contributions_per_contributor": st.column_config.BarChartColumn("Contributions distribution"),
            "num_prs": "Number of Pull Requests",
            "average_commit_message_length_to_main": "ACML* (main)",
            "average_commit_message_length": "ACML* (all)",
            "latest_commit": st.column_config.DatetimeColumn("Latest commit"),
        },
        hide_index=True,
    )
    st.text("*ACML = Average Commit Message Length")

    st.header("Content statistics")
    st.dataframe(
        df2,
        column_config={
            "group_nb": "Group Number",
            "num_docker_files": "Docker files",
            "num_workflow_files": "Workflow files",
            "has_requirement_file": "Requirement file",
            "has_makefile": "Makefile",
            "has_cloudbuild": "Cloudbuild",
            "repo_size": "Repository size",
            "readme_size": "Readme size",
            "using_dvc": "Using dvc",
            "warnings_raised": st.column_config.ProgressColumn(
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
