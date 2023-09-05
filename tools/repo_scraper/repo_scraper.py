import csv
import datetime
import os
import sys
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
GH_TOKEN = os.getenv("GH_TOKEN")
headers = {"Authorization": f"Bearer {GH_TOKEN}"}

if sys.platform == "win32":
    move_command = "move"
else:
    move_command = "mv"


def process_data(data: List[List[str]]):
    # remove empty emails
    new_data = [ ]
    for group in data:
        group[0] = int(group[0])  # convert group number to int
        new_data.append([group[0], len([g for g in group[1:-1] if g != ""]), group[-1]])
    return new_data


def load_data(filename: str) -> List[List[str]]:
    with open("latest_info.csv", "r") as f:
        csv_reader = csv.reader(f, delimiter=",")
        content = [ ]
        for row in csv_reader:
            content.append(row)

        header = content.pop(0)
        formatted_data = process_data(content)
    return formatted_data


def download_data(filename: str) -> None:
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
    split = repo.split("/")
    return f"{split[-2]}/{split[-1]}"


def get_default_branch_name(repo: str) -> str:
    response = requests.get(f"https://api.github.com/repos/{repo}", headers=headers)
    return response.json()["default_branch"]


def get_content(branch: str, url: str, repo: str, current_path: str) -> None:
    """
    Recursively download content from a github repo.

    Args:
    ----
        branch (str): branch name
        url (str): base url
        repo (str): repo name
        current_path (str): current path
    """
    response = requests.get(url, headers=headers)
    for file in response.json():
        if file["type"] == "dir":
            folder = file["name"]
            os.system(f"cd {current_path} & mkdir {folder}")
            get_content(branch, f"{url}/{folder}", repo, f"{current_path}/{folder}")
        else:
            path = file["path"]
            os.system(f"cd {current_path} & curl -s -OL https://raw.githubusercontent.com/{repo}/{branch}/{path}")


def get_content_recursive(url):
    all_content = [ ]
    content = requests.get(url, headers=headers).json()
    for c in content:
        if c['type'] == "dir":
            all_content += get_content_recursive(f"{url}/{c['name']}")
        else:
            all_content.append(c)
    return all_content


def write_to_file(filename, row, mode="a"):
    with open(filename, mode=mode, newline='') as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(row)


def main(out_folder="student_repos", download_content: bool = False):
    download_data("latest_info.csv")
    formatted_data = load_data("latest_info.csv")

    # loop for scraping the repository of each group
    os.makedirs(out_folder, exist_ok=True)

    if download_content:
        for group_nb, _, repo in formatted_data:
            os.system(f"cd {out_folder} && git clone {repo} && {move_command} {repo.split('/')[-1]} group_{group_nb}")

    else:
        write_to_file(
            "repo_data.csv",
            [
                "group_nb",
                "num_students",
                "num_contributors",
                "contributions_per_contributor",
                "num_prs",
                "average_commit_message_length_to_main",
                "average_commit_message_length",
                "num_docker_files",
                "num_workflow_files",
            ],
            mode="w"
        )

        for group_nb, num_students, repo in formatted_data:
            repo = reformat_repo(repo)
            exists = requests.get(f"https://api.github.com/repos/{repo}", headers=headers)
            if exists.status_code == 200:
                contributors = requests.get(f"https://api.github.com/repos/{repo}/contributors", headers=headers).json()
                contributors = {c['login']: {"contributions": c['contributions'], "commits_pr": 0} for c in contributors}
                num_contributors = len(contributors)

                prs = requests.get(
                    f"https://api.github.com/repos/{repo}/pulls", headers=headers, params={"state": "all", "per_page": 100}
                ).json()
                num_prs = len(prs)

                commits = requests.get(
                    f"https://api.github.com/repos/{repo}/commits",
                    headers=headers,
                    params={"state": "all", "per_page": 100}
                ).json()
                commit_messages = [c["commit"]["message"] for c in commits]
                average_commit_message_length_to_main = sum([len(c) for c in commit_messages]) / len(commit_messages)

                merged_prs = [p['number'] for p in prs if p['merged_at'] is not None]
                for pr_num in merged_prs:
                    pr_commits = requests.get(
                        f"https://api.github.com/repos/{repo}/pulls/{pr_num}/commits",
                        headers=headers,
                        params={"state": "all", "per_page": 100}
                    ).json()
                    commit_messages += [c["commit"]["message"] for c in pr_commits]
                    for comm in pr_commits:
                        if comm['committer'] is not None and comm['committer']['login'] in contributors:
                            contributors[comm['committer']['login']]['commits_pr'] += 1
                average_commit_message_length = sum([len(c) for c in commit_messages]) / len(commit_messages)

                contributions_per_contributor = [
                    c['contributions']+c['commits_pr'] for c in contributors.values()
                ]

                content = get_content_recursive(f"https://api.github.com/repos/{repo}/contents")
                docker_files = [c for c in content if c['name'] == "Dockerfile" or '.dockerfile' in c['name']]
                num_docker_files = len(docker_files)
                workflow_files = [c for c in content if c['path'].startswith(".github/workflows")]
                num_workflow_files = len(workflow_files)
            else:
                num_contributors = None
                num_prs = None
                average_commit_message_length_to_main = None
                average_commit_message_length = None
                contributions_per_contributor = None
                num_docker_files = None
                num_workflow_files = None


            write_to_file(
                "repo_data.csv",
                [
                    group_nb,
                    num_students,
                    num_contributors,
                    contributions_per_contributor,
                    num_prs,
                    average_commit_message_length_to_main,
                    average_commit_message_length,
                    num_docker_files,
                    num_workflow_files,
                ]
            )
        upload_data("repo_data.csv")

if __name__ == "__main__":
    main()

