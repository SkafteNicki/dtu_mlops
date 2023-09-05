import ast
import os
import sys

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

    # convert columns
    df['total_commits'] = df['contributions_per_contributor'].apply(
        lambda x: sum(ast.literal_eval(x)) if pd.notnull(x) else x
    )
    df['contributions_per_contributor'] = df['contributions_per_contributor'].apply(
        lambda x: ast.literal_eval(x) if pd.notnull(x) else x
    )

    # remove columns that are not needed
    df = df[
        [
            "group_nb",
            "num_students",
            "num_contributors",
            "total_commits",
            "contributions_per_contributor",
            "num_prs",
            "average_commit_message_length_to_main",
            "average_commit_message_length",
            "num_docker_files",
            "num_workflow_files",
            "has_requirement_file",
            "has_makefile",
        ]
    ]


    st.title("Group Github Stats")
    st.dataframe(
        df,
        column_config={
            "group_nb": "Group Number",
            "num_students": "Number of Students",
            "num_contributors": "Number of Contributors",
            "total_commits": "Total Commits",
            "contributions_per_contributor": st.column_config.BarChartColumn("Contributions distribution"),
            "num_prs": "Number of Pull Requests",
            "average_commit_message_length_to_main": "Average commit message length (main)",
            "average_commit_message_length": "Average commit message length (all)",
            "num_docker_files": "Number of docker files",
            "num_workflow_files": "Number of workflow files",
            "has_requirement_file": "Has requirement file",
            "has_makefile": "Has makefile",
        },
        hide_index=True,
    )

if __name__ == "__main__":
    main()
