"""
Basic script to scrape student repos and check a couple of properties.

Requires a student_info.txt file with the following format:
    group_nb,n_students,repo_url
For example:
    2	4	https://github.com/<user>/<repo-name>
And a github token in a token.txt file.
"""

import os

import click
import requests
import tqdm
from prettytable import PrettyTable

headers = {"Authorization": ""}

table = PrettyTable(
    field_names=[
        "Group nb",
        "Num students",
        "Readme file",
        "Num img files",
        "Num contributors",
        "Match number students",
        "Contribution stats",
        "PR count",
        "Questions missing",
    ]
)


def get_content(branch: str, url: str, repo: str, current_path: str) -> None:
    """Recursively download content from a github repo.

    Args:
        branch (str): branch name
        url (str): base url
        repo (str): repo name
        current_path (str): current path
    """
    response = requests.get(url, headers=headers, timeout=10)
    for file in response.json():
        if file["type"] == "dir":
            folder = file["name"]
            os.system(f"cd {current_path} & mkdir {folder}")
            get_content(branch, f"{url}/{folder}", repo, f"{current_path}/{folder}")
        else:
            path = file["path"]
            os.system(f"cd {current_path} & curl -s -OL https://raw.githubusercontent.com/{repo}/{branch}/{path}")


def default_branch(repo):
    """Get default branch of a repo."""
    response = requests.get(f"https://api.github.com/repos/{repo}", headers=headers, timeout=10)
    return response.json()["default_branch"]


@click.command()
@click.option("--out_folder", default="student_repos")
def main(out_folder):
    """Run scraper."""
    with open("student_info.txt", "r") as f:
        content = f.readlines()
    student_info = [c.split("\t") for c in content]
    student_info = [[c[0], c[1], c[2][19:-1]] for c in student_info]

    os.system(f"mkdir {out_folder}")
    for group_nb, _, repo in tqdm.tqdm(student_info):
        print(repo)
        # setup
        current_path = f"{out_folder}/{group_nb}"
        os.system(f"mkdir {current_path}")

        url = f"https://api.github.com/repos/{repo}/contents/reports"

        try:
            # find default branch
            branch = default_branch(repo)

            # download content
            get_content(branch, url, repo, current_path)
        except Exception:
            response = requests.get(f"https://api.github.com/repos/{repo}", headers=headers, timeout=10,)
            print(f"{group_nb} did not succeed with response {response}")

    for group_nb, n_students, repo in tqdm.tqdm(student_info):
        files = os.listdir(f"{group_nb}")
        nb_img_files = len(os.listdir(f"{group_nb}/figures")) if os.path.exists(f"{group_nb}/figures") else 0
        contributors = requests.get(
            f"https://api.github.com/repos/{repo}/contributors", headers=headers, timeout=10
        ).json()
        commits = [c["contributions"] for c in contributors]
        prs = len(
            requests.get(
                f"https://api.github.com/repos/{repo}/pulls",
                params={"state": "all", "per_page": 100},
                timeout=10,
            ).json()
        )

        if "README.md" in files:
            os.system(f"cd {group_nb} & python ../report.py html")
            os.system(f"cd {group_nb} & python ../report.py check > warnings.txt 2>&1")

            with open(f"{group_nb}/warnings.txt", "r") as file:
                content = file.readlines()
            nb_warnings = len([c for c in content if "TeacherWarning" in c])
        else:
            nb_warnings = "-"

        table.add_row(
            [
                group_nb,
                n_students,
                "README.md" in files,
                nb_img_files,
                len(contributors),
                int(n_students) == len(contributors),
                f"{commits}={sum(commits)}",
                prs,
                nb_warnings,
            ]
        )

    print(table)

    with open("table.csv", "w", newline="") as f_output:
        f_output.write(table.get_csv_string())


if __name__ == "__main__":
    main()
