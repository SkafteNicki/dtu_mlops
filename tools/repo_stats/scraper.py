import csv
import datetime
import json
import logging
import os
from pathlib import Path

import requests
from dotenv import load_dotenv
from google.cloud.storage import Client
from models import GroupInfo, RepoContent, Report, RepoStats
from typer import Typer

load_dotenv()

GH_TOKEN = os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN")
headers = {"Authorization": f"Bearer {GH_TOKEN}"}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def upload_data(file_name: str) -> None:
    """Uploads the repo stats data to GCS."""
    storage_client = Client()
    bucket = storage_client.bucket("mlops_group_repository")
    blob = bucket.blob(file_name)
    blob.upload_from_filename(file_name)


def download_data(file_name: str) -> None:
    """Downloads the group-repository data from GCS."""
    storage_client = Client()
    bucket = storage_client.bucket("mlops_group_repository")
    blob = bucket.blob(file_name)
    blob.download_to_filename(file_name)


def load_data(file_name: str) -> list[GroupInfo]:
    """Loads the group-repository data into a DataFrame."""
    with Path(file_name).open() as f:
        csv_reader = csv.reader(f, delimiter=",")
        content = []
        for i, row in enumerate(csv_reader):
            if i == 0:  # Skip the header
                continue
            group = GroupInfo(
                group_number=int(row[0]),
                student_1=row[1] if row[1] != "" else None,
                student_2=row[2] if row[2] != "" else None,
                student_3=row[3] if row[3] != "" else None,
                student_4=row[4] if row[4] != "" else None,
                student_5=row[5] if row[5] != "" else None,
                repo_url=row[6],
            )
            content.append(group)
    return content


app = Typer()


@app.command()
def main():
    """Main function to scrape the group-repository data."""
    logger.info("Getting group-repository information")
    if "group_info.csv" not in os.listdir():
        download_data("group_info.csv")
    group_data = load_data("group_info.csv")
    logger.info("Group-repository information loaded successfully")

    repo_stats: list[RepoContent] = []
    for index, group in enumerate(group_data):
        logger.info(f"Processing group {group.group_number}, {index+1}/{len(group_data)}")

        if group.repo_accessible:
            contributors = group.contributors
            num_contributors = len(contributors)

            prs = group.prs
            num_prs = len(prs)

            commits = group.commits
            num_commits_to_main = len(commits)
            commit_messages = [c["commit"]["message"] for c in commits]
            average_commit_length_to_main = sum([len(c) for c in commit_messages]) / len(commit_messages)
            latest_commit = commits[0]["commit"]["author"]["date"]

            merged_prs = [p["number"] for p in prs if p["merged_at"] is not None]
            for pr_num in merged_prs:
                pr_commits = requests.get(
                    f"{group.repo_api}/pulls/{pr_num}/commits", headers=headers, timeout=100
                ).json()
                commit_messages += [c["commit"]["message"] for c in pr_commits]
                for commit in pr_commits:
                    for contributor in contributors:
                        if (
                            commit["committer"] is not None
                            and "login" in commit["committer"]
                            and contributor.login == commit["author"]["login"]
                        ):
                            contributor.commits_pr += 1
            average_commit_length = sum([len(c) for c in commit_messages]) / len(commit_messages)

            contributions_per_contributor = [c.total_commits for c in contributors]
            total_commits = sum(contributions_per_contributor)

            repo_content = RepoContent(
                group_number=group.group_number, repo_api=group.repo_api, default_branch=group.default_branch
            )
            num_docker_files = repo_content.num_docker_files
            num_python_files = repo_content.num_python_files
            num_workflow_files = repo_content.num_workflow_files
            has_requirements_file = repo_content.has_requirements_file
            has_cloudbuild = repo_content.has_cloudbuild
            using_dvc = repo_content.using_dvc
            repo_size = repo_content.repo_size
            readme_length = repo_content.readme_length

            report = Report(
                group_number=group.group_number, repo_api=group.repo_api, default_branch=group.default_branch
            )
            num_warnings = report.check_answers

        else:
            num_contributors = None
            num_prs = None
            num_commits_to_main = None
            average_commit_length_to_main = None
            latest_commit = None
            average_commit_length = None
            total_commits = None
            contributions_per_contributor = None
            total_commits = None

            num_docker_files = None
            num_python_files = None
            num_workflow_files = None
            has_requirements_file = None
            has_cloudbuild = None
            using_dvc = None
            repo_size = None
            readme_length = None

            num_warnings = None

        repo_stat = RepoStats(
            group_number=group.group_number,
            group_size=group.group_size,
            num_contributors=num_contributors,
            num_prs=num_prs,
            num_commits_to_main=num_commits_to_main,
            average_commit_length_to_main=average_commit_length_to_main,
            latest_commit=latest_commit,
            average_commit_length=average_commit_length,
            contributions_per_contributor=contributions_per_contributor,
            total_commits=total_commits,
            num_docker_files=num_docker_files,
            num_python_files=num_python_files,
            num_workflow_files=num_workflow_files,
            has_requirements_file=has_requirements_file,
            has_cloudbuild=has_cloudbuild,
            using_dvc=using_dvc,
            repo_size=repo_size,
            readme_length=readme_length,
            num_warnings=num_warnings,
        )
        repo_stats.append(repo_stat)

    logger.info("Writing repo stats to file")
    filename = f"repo_stats{datetime.datetime.now(tz=datetime.UTC).isoformat(timespec='seconds')}.json"
    with open("repo_stats.json", "w") as f:
        json.dump([r.model_dump() for r in repo_stats])
    with open(filename, "w") as f:
        json.dump([r.model_dump() for r in repo_stats])

    logger.info("Uploading repo stats to GCS")
    upload_data("repo_stats.json")
    upload_data(filename)

    logger.info("Cleaning locally temp files")
    Path("README.md").unlink()
    Path("report.py").unlink()


if __name__ == "__main__":
    app()
