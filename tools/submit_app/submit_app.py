"""Simple submission streamlit application.

Run locally with (from root folder)
    streamlit run tools/submit_app/submit_app.py
"""

import csv
import datetime
import os
import sys

import dropbox
import streamlit as st
from dotenv import load_dotenv
from dropbox.exceptions import ApiError, AuthError

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

DEFAULT_EMAIL = "sXXXXXX@student.dtu.dk"


def send_to_dropbox_and_get_group_nb(
    github_repo: str,
    student1: str,
    student2: str,
    student3: str,
    student4: str,
    student5: str,
) -> int:
    """Send the group information to dropbox and return the next group number."""
    fields = ["group nb", "student 1", "student 2", "student 3", "student 4", "student 5", "github_repo"]

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

        start_over = False
        try:
            dbx.files_download_to_file("latest_info.csv", "/latest_info.csv")
        except ApiError:
            print("latest file not found, creating new file")
            start_over = True

        if start_over:
            with open("info.csv", "w", newline="") as f:
                csv_writer = csv.writer(f, delimiter=",")
                csv_writer.writerow(fields)
            file_to_upload = "info.csv"
        else:
            with open("latest_info.csv") as f:
                csv_reader = csv.reader(f, delimiter=",")
                content = []
                for row in csv_reader:
                    print(row)
                    content.append(row)
                if len(content) == 1:  # header only
                    group_nb = 0
                else:
                    group_nb = int(content[-1][0])

                new_group_nb = group_nb + 1

            with open("latest_info.csv", "a", newline="") as f:
                csv_writer = csv.writer(f, delimiter=",")
                csv_writer.writerow([new_group_nb, student1, student2, student3, student4, student5, github_repo])
            file_to_upload = "latest_info.csv"

        now = datetime.datetime.now(tz=datetime.UTC).strftime("%Y_%m_%d_%H_%M_%S")
        with open(file_to_upload, "rb") as f:
            dbx.files_upload(f.read(), f"/{now}_info.csv")
        with open(file_to_upload, "rb") as f:
            dbx.files_upload(f.read(), "/latest_info.csv", mode=dropbox.files.WriteMode.overwrite)

        if start_over:
            # if we started over we redo everything to get the group number
            new_group_nb = send_to_dropbox_and_get_group_nb(
                github_repo,
                student1,
                student2,
                student3,
                student4,
                student5,
            )

        return new_group_nb


def validate_text_input(
    student1: str,
    student2: str,
    student3: str,
    student4: str,
    student5: str,
    github_repo: str,
) -> bool:
    """
    Validate input from text fields.

    Checks:
    - not all student emails are default
    - at least three student emails
    - GitHub repo is valid
    """
    emails = [student1, student2, student3, student4, student5]
    if all(email == DEFAULT_EMAIL for email in emails):
        st.error("Please enter at least two student emails!")
        return False
    data = [email for email in emails if email != DEFAULT_EMAIL]
    if len(data) < 3:
        st.error("Please enter at least three student emails! Minimum group size is 3.")
        return False
    if "https://github.com/" not in github_repo:
        st.error("Please enter a valid GitHub repo!")
        return False
    return True


def main() -> None:
    """Streamlit application submission form."""
    with st.columns([1, 8, 1])[1]:
        st.title("DTU course 02476 MLOps")
        st.header("Group Information")

        student1 = st.text_input(
            "Student 1",
            value=DEFAULT_EMAIL,
            max_chars=22,
            key="student1",
        )

        student2 = st.text_input(
            "Student 2",
            value=DEFAULT_EMAIL,
            max_chars=22,
            key="student2",
        )

        student3 = st.text_input(
            "Student 3",
            value=DEFAULT_EMAIL,
            max_chars=22,
            key="student3",
        )

        student4 = st.text_input(
            "Student 4",
            value=DEFAULT_EMAIL,
            max_chars=22,
            key="student4",
        )

        student5 = st.text_input(
            "Student 5",
            value=DEFAULT_EMAIL,
            max_chars=22,
            key="student5",
        )

        github_repo = st.text_input(
            "Github Repo",
            value="https://github.com/Username/reponame",
            key="github_repo",
        )

        st.write("Check that everything is correct before submitting!")

        button = st.empty()
        isclicked = button.button("Submit!")
        if isclicked:
            if validate_text_input(student1, student2, student3, student4, student5, github_repo):
                button.empty()

                try:
                    group_nb = send_to_dropbox_and_get_group_nb(
                        github_repo,
                        student1=student1 if student1 != DEFAULT_EMAIL else "",
                        student2=student2 if student2 != DEFAULT_EMAIL else "",
                        student3=student3 if student3 != DEFAULT_EMAIL else "",
                        student4=student4 if student4 != DEFAULT_EMAIL else "",
                        student5=student5 if student5 != DEFAULT_EMAIL else "",
                    )
                except:  # noqa: E722
                    st.error(
                        "The application did not manage to send your data. Please try again later."
                        "If the problem persists, contact the course responsible.",
                    )
                    return

                st.write("Group information submitted!")
                st.header("Your group number is: " + str(group_nb))
                st.write("Remember to write this down!")
            else:
                pass


if __name__ == "__main__":
    main()
