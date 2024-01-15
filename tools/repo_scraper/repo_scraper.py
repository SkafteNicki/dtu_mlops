"""Tool for scraping student repos for information.

Run locally with (from root folder)
    python tools/repo_scraper/repo_scraper.py
"""

import csv
import datetime
import os
import shutil
import sys
from pathlib import Path
from subprocess import PIPE, Popen
from typing import List

import dropbox
import requests
from dotenv import load_dotenv
from dropbox.exceptions import AuthError

load_dotenv()
DROPBOX_TOKEN = os.getenv("DROPBOX_TOKEN")
DROPBOX_APP_KEY = os.getenv("DROPBOX_APP_KEY")
DROPBOX_APP_SECRET = os.getenv("DROPBOX_APP_SECRET")
DROPBOX_REFRESH_TOKEN = os.getenv("DROPBOX_REFRESH_TOKEN")
GH_TOKEN = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
headers = {"Authorization": f"Bearer {GH_TOKEN}"}


def process_data(data: List[List[str]]):
    """Process the data from the csv file."""
    # remove empty emails
    new_data = []
    for group in data:
        group[0] = int(group[0])  # convert group number to int
        new_data.append([group[0], len([g for g in group[1:-1] if g != ""]), group[-1]])
    return new_data


def load_data(filename: str) -> List[List[str]]:
    """Load the data from the csv file."""
    with open("latest_info.csv", "r") as f:
        csv_reader = csv.reader(f, delimiter=",")
        content = []
        for row in csv_reader:
            content.append(row)

        header = content.pop(0)
        formatted_data = process_data(content)
    return formatted_data


def download_data(filename: str) -> None:
    """Download specific file from dropbox."""
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


def upload_data(filename: str) -> None:
    """Upload specific file to dropbox."""
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

        now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        with open(filename, "rb") as f:
            dbx.files_upload(f.read(), f"/{now}_{filename}")
        with open(filename, "rb") as f:
            dbx.files_upload(f.read(), f"/latest_{filename}", mode=dropbox.files.WriteMode.overwrite)


def reformat_repo(repo: str):
    """Extract from the url the user id and repository name only."""
    split = repo.split("/")
    return f"{split[-2]}/{split[-1]}"


def get_default_branch_name(repo: str) -> str:
    """Get the default branch name of a github repo."""
    response = requests.get(f"https://api.github.com/repos/{repo}", headers=headers, timeout=100)
    return response.json()["default_branch"]


def get_content(branch: str, url: str, repo: str, current_path: str) -> None:
    """Recursively download content from a github repo."""
    response = requests.get(url, headers=headers, timeout=100)
    for file in response.json():
        if file["type"] == "dir":
            folder = file["name"]
            os.system(f"cd {current_path} & mkdir {folder}")
            get_content(branch, f"{url}/{folder}", repo, f"{current_path}/{folder}")
        else:
            path = file["path"]
            os.system(f"cd {current_path} & curl -s -OL https://raw.githubusercontent.com/{repo}/{branch}/{path}")


def get_content_recursive(url):
    """Extract all content from a github repo recursively."""
    all_content = []
    content = requests.get(url, headers=headers, timeout=10).json()
    for c in content:
        if c["type"] == "dir":
            all_content += get_content_recursive(f"{url}/{c['name']}")
        else:
            all_content.append(c)
    return all_content


def write_to_file(filename, row, mode="a"):
    """Write to a local csv file."""
    with open(filename, mode=mode, newline="") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(row)


def main(
    out_folder: str = "student_repos",
    timeout_clone: str = "2m",
):
    """Extract group statistics from github."""
    print("Getting the repository information")
    if "latest_info.csv" not in os.listdir():
        download_data("latest_info.csv")
    formatted_data = load_data("latest_info.csv")

    # loop for scraping the repository of each group
    print("Cleaning out old data if needed")
    if os.path.isdir(out_folder):  # non-empty folder, delete content
        shutil.rmtree(out_folder)
    os.makedirs(out_folder)

    # clone repos
    print("====== Cloning repos ======")
    for index, data in enumerate(formatted_data):
        group_nb, _, repo = data
        print(f"Cloning group {group_nb}, {index}/{len(formatted_data)}")
        out = os.system(f"cd {out_folder} && timeout -v {timeout_clone} git clone -q {repo}")
        clone_succes = out == 0
        folder_name = repo.split("/")[-1]
        if clone_succes:
            os.system(f"cd {out_folder} && cp -r {folder_name} group_{group_nb} && rm -rf {folder_name}")
        else:
            if folder_name in os.listdir(out_folder):
                shutil.rmtree(f"{out_folder}/{folder_name}")
        data.append(clone_succes)

    # create file for data
    write_to_file(
        "repo_data.csv",
        [
            "group_nb",
            "num_students",
            "num_contributors",
            "num_prs",
            "average_commit_message_length_to_main",
            "latest_commit",
            "average_commit_message_length",
            "contributions_per_contributor",
            "total_commits",
            "num_docker_files",
            "num_workflow_files",
            "has_requirement_file",
            "has_makefile",
            "has_cloudbuild",
            "repo_size",
            "readme_size",
            "using_dvc",
            "warnings_raised",
        ],
        mode="w",
    )

    # extract info through API
    print("====== Extracting info through API ======")
    for index, (group_nb, num_students, repo, clone_succes) in enumerate(formatted_data):
        print(f"Processing group {group_nb}, {index}/{len(formatted_data)}")
        repo = reformat_repo(repo)
        exists = requests.get(f"https://api.github.com/repos/{repo}", headers=headers, timeout=100)
        if exists.status_code == 200:
            contributors = requests.get(
                f"https://api.github.com/repos/{repo}/contributors", headers=headers, timeout=100
            ).json()
            contributors = {c["login"]: {"contributions": c["contributions"], "commits_pr": 0} for c in contributors}
            num_contributors = len(contributors)

            prs = requests.get(
                f"https://api.github.com/repos/{repo}/pulls",
                headers=headers,
                params={"state": "all", "per_page": 100},
                timeout=100,
            ).json()
            num_prs = len(prs)

            commits = requests.get(
                f"https://api.github.com/repos/{repo}/commits",
                headers=headers,
                params={"state": "all", "per_page": 100},
                timeout=100,
            ).json()
            commit_messages = [c["commit"]["message"] for c in commits]
            average_commit_message_length_to_main = sum([len(c) for c in commit_messages]) / len(commit_messages)
            latest_commit = commits[0]["commit"]["author"]["date"]

            merged_prs = [p["number"] for p in prs if p["merged_at"] is not None]
            for pr_num in merged_prs:
                pr_commits = requests.get(
                    f"https://api.github.com/repos/{repo}/pulls/{pr_num}/commits",
                    headers=headers,
                    params={"state": "all", "per_page": 100},
                    timeout=100,
                ).json()
                commit_messages += [c["commit"]["message"] for c in pr_commits]
                for comm in pr_commits:
                    if comm["committer"] is not None and comm["committer"]["login"] in contributors:
                        contributors[comm["committer"]["login"]]["commits_pr"] += 1
            average_commit_message_length = sum([len(c) for c in commit_messages]) / len(commit_messages)

            contributions_per_contributor = [c["contributions"] + c["commits_pr"] for c in contributors.values()]
            total_commits = sum(contributions_per_contributor)

            content = get_content_recursive(f"https://api.github.com/repos/{repo}/contents")
            docker_files = [c for c in content if c["name"] == "Dockerfile" or ".dockerfile" in c["name"]]
            num_docker_files = len(docker_files)
            workflow_files = [c for c in content if c["path"].startswith(".github/workflows")]
            num_workflow_files = len(workflow_files)
            has_requirement_file = len([c for c in content if c["name"] == "requirements.txt"]) > 0
            has_makefile = len([c for c in content if c["name"] == "Makefile"]) > 0
            has_cloudbuild = len([c for c in content if "cloudbuild.yaml" in c["name"]]) > 0
        else:
            num_contributors = None
            num_prs = None
            average_commit_message_length_to_main = None
            latest_commit = None
            average_commit_message_length = None
            contributions_per_contributor = None
            total_commits = None
            num_docker_files = None
            num_workflow_files = None
            has_requirement_file = None
            has_makefile = None
            has_cloudbuild = None

        if clone_succes:
            path = Path(f"{out_folder}/group_{group_nb}")
            repo_size = sum([f.stat().st_size for f in path.glob("**/*") if f.is_file()]) / 1_048_576  # in MB

            if "README.md" in os.listdir(f"{out_folder}/group_{group_nb}"):
                with open(f"{out_folder}/group_{group_nb}/README.md", "r") as f:
                    content = f.read()
                readme_size = len(content.split(" "))
            else:
                readme_size = None

            using_dvc = ".dvc" in os.listdir(f"{out_folder}/group_{group_nb}")

            warnings_raised = None
            if "reports" in os.listdir(f"{out_folder}/group_{group_nb}"):
                report_dir = os.listdir(f"{out_folder}/group_{group_nb}/reports")
                if "README.md" in report_dir and "report.py" in report_dir:
                    p = Popen(
                        ["python", "report.py", "check"],
                        cwd=f"{out_folder}/group_{group_nb}/reports",
                        stdout=PIPE,
                        stderr=PIPE,
                        stdin=PIPE,
                    )
                    output = p.stderr.read()
                    warnings_raised = len(output.decode("utf-8").split("\n")[:-1:2])
        else:
            repo_size = None
            readme_size = None
            using_dvc = None
            warnings_raised = None

        write_to_file(
            "repo_data.csv",
            [
                group_nb,
                num_students,
                num_contributors,
                num_prs,
                average_commit_message_length_to_main,
                latest_commit,
                average_commit_message_length,
                contributions_per_contributor,
                total_commits,
                num_docker_files,
                num_workflow_files,
                has_requirement_file,
                has_makefile,
                has_cloudbuild,
                repo_size,
                readme_size,
                using_dvc,
                warnings_raised,
            ],
            mode="a",
        )
    upload_data("repo_data.csv")


if __name__ == "__main__":
    main()
