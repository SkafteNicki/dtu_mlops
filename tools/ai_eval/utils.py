import csv
import os
from pathlib import Path

from google.cloud.storage import Client
from loguru import logger
from models import GroupInfo, RepoMix


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


def get_data() -> list[GroupInfo]:
    """Get the group-repository information."""
    logger.info("Getting group-repository information")
    if "group_info.csv" not in os.listdir():
        download_data("group_info.csv")
    group_data = load_data("group_info.csv")
    logger.info("Group-repository information loaded successfully")
    return group_data


def call_repomix(repo: str, repomix_config: RepoMix, out_folder: str = "output") -> None:
    """Call repomix on a repository."""
    repomix_config.dump_json("repomix.config.json")
    logger.info(f"Running repomix on {repo}")
    current_dir = os.getcwd()
    os.system(f"repomix -c {current_dir}/repomix.config.json --remote {repo} --verbose >> output.log")
    repo_name = "_".join(repo.split("/")[-2:])
    os.system(f"mkdir --parents  {out_folder}/{repo_name}")
    os.system(f"mv output.log {out_folder}/{repo_name}/output.log")
    os.system(f"mv repomix-output.md {out_folder}/{repo_name}/repomix-output.md")
    os.system("rm repomix.config.json")


def get_repo_content(repository: str, repomix_config: RepoMix) -> str:
    """Get the code from a repository."""
    if repository.startswith("https://github.com"):
        call_repomix(repository, repomix_config, out_folder="output")
        repo_name = "_".join(repository.split("/")[-2:])
        path = Path(f"output/{repo_name}/repomix-output.md")
    else:
        path = Path(repository)
    with path.open("r") as file:
        return file.read()
