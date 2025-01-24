import json
import os
import shutil
from pathlib import Path

import logfire
import typer
from devtools import pprint
from dotenv import load_dotenv
from loguru import logger
from models import GroupInfo, RepoMix, TACodeResponse, TADependency, TAReportResponse
from openai import AsyncAzureOpenAI
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from utils import get_data, get_repo_content

load_dotenv(".env")


logfire.configure()
logfire.instrument_asyncpg()

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)
client = AsyncAzureOpenAI(
    azure_endpoint=os.environ.get("AZURE_ENDPOINT"),
    api_version=os.environ.get("API_VERSION"),
    api_key=os.environ.get("API_KEY"),
)
model = OpenAIModel(os.environ.get("MODEL"), openai_client=client)


def finalize(responses: list, clean: bool = True, name: str = "responses.json") -> None:
    """Save responses and clean up if needed."""
    with open(name, "w") as f:  # Save responses in case of error
        json.dump([response.model_dump() for response in responses], f, indent=4)
    if clean:
        shutil.rmtree(Path("output"))


@app.command()
def codebase(group_nb: None | int = None, clean: bool = True) -> None:
    """Main function to evaluate the codebase of a group."""
    ta_agent = Agent(
        model=model,
        deps_type=GroupInfo,
        result_type=TACodeResponse,
        system_prompt="""
        You are a teaching assistant for a university level course on machine learning operations.
        Your job is to review and evaluate the students final group project. Review the code for adherence to coding
        best practices and industry standards. Identify areas where the code could be improved in terms of readability,
        maintainability and efficiency. You need to provide the following feedback to the students.

        * Score the code quality on a scale from 1 to 5 using these criteria:
            1: Poor - The code violates many best practices, has poor readability, and is difficult to maintain.
            2: Below Average - The code has significant issues in readability, maintainability, or efficiency.
            3: Average - The code meets basic standards but has room for improvement.\n
            4: Good - The code adheres to most best practices with only minor issues.\n
            5: Excellent - The code is clean, maintainable, and follows industry best practices.\n

        * Score the unit testing in the codebase using these criteria:

            1: Poor - Minimal or no unit tests, with negligible code coverage.\n
            2: Below Average - Unit tests exist but fail to cover significant parts of the codebase.\n
            3: Average - Adequate unit test coverage but with noticeable gaps.\n
            4: Good - Unit tests cover most of the critical functionality with minor areas lacking.\n
            5: Excellent - Comprehensive unit tests thoroughly cover the entire codebase.\n

        * Score the continuous integration and deployment process using these criteria:

            1: Poor - No CI/CD pipeline or the pipeline is broken.\n
            2: Below Average - CI/CD pipeline exists but is unreliable or lacks automation.\n
            3: Average - CI/CD pipeline is functional but lacks some best practices.\n
            4: Good - CI/CD pipeline is reliable and automated with only minor issues.\n
            5: Excellent - CI/CD pipeline is robust, automated, and follows industry best practices.\n

        * Provide a brief summary of the code quality, unit testing, ci/cd, along with any suggestions for improvement.
            You should be using no more than 500 words for this summary.

        * Finally, provide a both a overall score from 1-10 for the hole codebase how well it implements a machine
            learning operations pipeline and also provide your confidence in a score from 1-10
        """,
    )

    @ta_agent.system_prompt
    async def add_group_information(ctx: RunContext[GroupInfo]) -> str:
        group_number = ctx.deps.group_info.group_number
        repo_content = get_repo_content(ctx.deps.group_info.repo_url, ctx.deps.repomix)
        return f"""
        Group {group_number} has submitted the following repository:
        {repo_content}
        """

    group_data = get_data()
    if group_nb:
        group_data = [group_data[group_nb - 1]]

    responses: list[TACodeResponse] = []
    for group in group_data:
        logger.info(f"Processing group {group.group_number}")
        deps = TADependency(
            group_info=group,
            repomix=RepoMix(
                ignore=RepoMix.Ignore(
                    customPatterns=[
                        ".dvc/*",
                        "*.dvc",
                        "**/*/report.py",
                        "reports/README.md",
                        "*.gitignore",
                        "**/*.ipynb",
                        "**/*.html",
                        "uv.lock",
                        "data/**",
                        "**/*.csv",
                        "log/**",
                        "logs/**",
                        "outputs/**",
                    ]
                )
            ),
        )
        try:
            result = ta_agent.run_sync("What do you think of the groups repository?", deps=deps)
            result.data.request_usage = result.usage()
            pprint(result.data)
            responses.append(result.data)
        except Exception as e:
            finalize(responses, clean, name="codebase")
            raise e
    finalize(responses, clean, name="codebase")


@app.command()
def report(group_nb: None | int = None, clean: bool = True) -> None:
    """Main function to evaluate the report of a group."""
    ta_agent = Agent(
        model=model,
        deps_type=GroupInfo,
        result_type=TAReportResponse,
        system_prompt="""
        You are a teaching assistant for a university level course on machine learning operations. You are tasked with
        correcting a student's report which is provided in markdown format. The report is a template consisting of 31
        questions and are fologrmatted into a couple of sections:  Group information, Coding environment, Version
        control, Running code and tracking experiments, Working in the cloud, Deployment, Overall discussion of project.
        Additionally, it contains a checklist of 52 items that needs to be filled out. For each of the sections
        (except Group information), you will provide a brief summary of the student's response and then provide feedback
        on the accuracy and completeness of the response. You will also provide suggestions for improvement. Score each
        section on a scale from 1 to 5 based on the following criteria:
        1: Poor - The response is inaccurate, incomplete, or contains significant errors.
        2: Below Average - The response is partially accurate but contains several errors or omissions.
        3: Average - The response is mostly accurate but contains minor errors or omissions.
        4: Good - The response is accurate and complete with only minor issues.
        5: Excellent - The response is accurate, complete, and well-reasoned.
        You should penelise the students for not answering questions. Only focus on the students answers. In addition
        you need to return how many of the 52 items from the checklist were completed. Provide also a summary of the
        overall report evaluation using no more than 300 words. Finally, return a grading score from 1-10 and your
        confidence in the grading from 1-10.
        """,
    )

    @ta_agent.system_prompt
    async def add_group_information(ctx: RunContext[GroupInfo]) -> str:
        group_number = ctx.deps.group_info.group_number
        repo_content = get_repo_content(ctx.deps.group_info.repo_url, ctx.deps.repomix)
        return f"""
        Group {group_number} has submitted the following report:
        {repo_content}
        """

    group_data = get_data()
    if group_nb:
        group_data = [group_data[group_nb - 1]]

    responses: list[TAReportResponse] = []
    for group in group_data:
        deps = TADependency(group_info=group, repomix=RepoMix(include=["reports/README.md"]))
        try:
            result = ta_agent.run_sync("What do you think of the groups report?", deps=deps)
            result.data.request_usage = result.usage()
            pprint(result.data)
            responses.append(result.data)
        except Exception as e:
            finalize(responses, clean, name="codebase")
            raise e
    finalize(responses, clean, name="codebase")


if __name__ == "__main__":
    app()
