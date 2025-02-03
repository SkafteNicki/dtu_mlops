import csv
import os
import shutil
import time
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import typer
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from google.cloud import storage
from playwright.sync_api import sync_playwright

load_dotenv()

USER = os.getenv("USER")
PASSWORD = os.getenv("PASSWORD")


def extract_datetime(folder_name):
    """Extracts the datetime part from the folder name."""
    datetime_part = folder_name.split(" - ")[-1]
    datetime_part = datetime_part.replace("AM", " AM").replace("PM", " PM")
    return datetime.strptime(datetime_part, "%d %B, %Y %I%M %p")  # noqa: DTZ007


def extract_base_github_url(url):
    """Extracts the base URL of a GitHub repo, removing extra components like branch or file paths, '.git' suffix."""
    parsed = urlparse(url)
    path_parts = parsed.path.split("/")

    # Ensure the path contains at least 'owner' and 'repo' components
    if len(path_parts) >= 3:
        owner = path_parts[1]
        repo = path_parts[2]
        # Remove '.git' suffix if present
        if repo.endswith(".git"):
            repo = repo[:-4]
        # Construct the base GitHub URL
        return f"https://{parsed.netloc}/{owner}/{repo}"
    msg = "Invalid GitHub URL format."
    raise ValueError(msg)


def download_from_learn(course: str) -> tuple[str, str]:
    """Download the group membership and project repositories from DTU Learn."""
    with sync_playwright() as p:
        browser = p.firefox.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        page.goto(f"https://learn.inside.dtu.dk/d2l/home/{course}")
        page.get_by_placeholder("User").fill(USER)
        page.get_by_placeholder("Password").fill(PASSWORD)
        page.get_by_role("button", name="Sign in").click()
        page.get_by_role("button", name="My Course").click()
        page.get_by_role("link", name="Grades").click()
        page.get_by_role("link", name="Enter Grades").click()
        page.get_by_role("button", name="Export").click()
        page.get_by_label("Both").check()
        page.get_by_role("button", name="Export to CSV").click()
        with page.expect_download() as download1_info:
            page.get_by_role("button", name="Download").click()
        download1 = download1_info.value
        download1.save_as(os.path.join(os.getcwd(), download1.suggested_filename))
        page.get_by_role("button", name="Close").click()
        page.get_by_role("button", name="Cancel").click()
        page.get_by_role("link", name="Assignments").click()
        page.get_by_role("link", name="Project repository link").click()
        page.get_by_role("checkbox", name="Select all rows").check()
        page.get_by_role("button", name="Download").click()
        time.sleep(2)  # for some reason, the download doesn't work without this delay
        with page.expect_download() as download2_info:
            page.get_by_role("button", name="Download").click()
        download2 = download2_info.value
        download2.save_as(os.path.join(os.getcwd(), download2.suggested_filename))
        page.get_by_role("button", name="Close").click()
        context.close()
        browser.close()
    return download1.suggested_filename, download2.suggested_filename


def create_grouped_csv(download1: str) -> None:
    """Create a grouped CSV file from the downloaded group membership."""
    groups = defaultdict(list)
    with open(download1, encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            group = row["Project Groups"]
            if group:  # Only consider rows with a project group
                # Extract the student ID, removing the '#' if present
                username = row["Username"].lstrip("#")
                groups[group.strip("MLOPS ")].append(username)

    # Sort groups by numeric value of group number
    sorted_groups = sorted(groups.items(), key=lambda x: int(x[0].split()[-1]))

    # Write the transformed data
    with open("grouped_students.csv", mode="w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["group_nb", "student 1", "student 2", "student 3", "student 4", "student 5"])
        for group, students in sorted_groups:
            row = [group] + students[:5] + [""] * (5 - len(students[:5]))
            writer.writerow(row)


def unzip_assignments_and_extract_links(download2: str) -> dict:
    """Unzip the downloaded assignments and extract the repository links."""
    os.makedirs("extracted_files", exist_ok=True)
    with zipfile.ZipFile(download2, "r") as zip_ref:
        zip_ref.extractall("extracted_files")

    folders = Path("extracted_files").iterdir()
    grouped_folders = defaultdict(list)
    for folder in folders:
        if folder.is_dir():
            group_key = folder.name.split(" - ")[1].strip("MLOPS ")
            grouped_folders[group_key].append(folder)

    # Get the most recent folder for each group
    most_recent_folders = {}
    for group_key, group_folders in grouped_folders.items():
        most_recent_folder = max(group_folders, key=lambda x: extract_datetime(x.name))
        most_recent_folders[group_key] = most_recent_folder

    group_links = {}
    for group_number, folder in most_recent_folders.items():
        if folder.is_dir():
            for file in folder.iterdir():
                if file.suffix == ".html":
                    with open(file, encoding="utf-8") as html_file:
                        soup = BeautifulSoup(html_file, "html.parser")
                        link_tag = soup.find("a", href=True)
                        if link_tag:
                            link_tag["href"] = extract_base_github_url(link_tag["href"])
                            group_links[group_number] = link_tag["href"]
    return group_links


def main(
    course: str = typer.Argument(..., help="The course code"),
    clean: bool = typer.Option(True, help="Clean the extracted files"),
    upload: bool = typer.Option(False, help="Upload the updated CSV file to GCS"),
) -> None:
    """Automatically download group membership and project repositories from DTU Learn."""
    download1, download2 = download_from_learn(course)

    create_grouped_csv(download1)
    group_links = unzip_assignments_and_extract_links(download2)

    grouped_csv_path = "grouped_students.csv"
    updated_csv_path = "group_info.csv"
    total_students = 0
    total_groups = 0

    with (
        open(grouped_csv_path, encoding="utf-8") as infile,
        open(updated_csv_path, mode="w", encoding="utf-8", newline="") as outfile,
    ):
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # Add a new column header
        headers = next(reader)
        headers.append("github_repo")
        writer.writerow(headers)

        # Update rows with links
        for row in reader:
            total_students += sum(1 for student in row[1:6] if student)
            group_number = row[0].split()[-1]  # Extract the numeric part of group number
            repo_link = group_links.get(group_number, "No Link Found")
            row.append(repo_link)
            writer.writerow(row)

            total_groups += 1

    print(f"Updated CSV file saved to: {updated_csv_path}")

    # Print totals
    print(f"Total number of students: {total_students}")
    print(f"Total number of groups: {total_groups}")

    if clean:
        shutil.rmtree(Path("extracted_files"))
        Path("grouped_students.csv").unlink()
        Path(download1).unlink()
        Path(download2).unlink()

    if upload:
        storage_client = storage.Client()
        bucket = storage_client.bucket("mlops_group_repository")
        blob = bucket.blob("group_info.csv")
        blob.upload_from_filename(updated_csv_path)
        print("Updated CSV file uploaded to GCS")


if __name__ == "__main__":
    typer.run(main)
