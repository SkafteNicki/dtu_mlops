import json
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_ai.usage import Usage


class RepoMix(BaseModel):
    """Configuration for the repomix."""

    class Output(BaseModel):
        """Output configuration for the repomix."""

        filePath: str = "repomix-output.md"  # noqa: N815
        style: str = "markdown"
        removeComments: bool = False  # noqa: N815
        removeEmptyLines: bool = False  # noqa: N815
        showLineNumbers: bool = False  # noqa: N815
        copyToClipboard: bool = False  # noqa: N815
        topFilesLength: int = 10  # noqa: N815

    class Ignore(BaseModel):
        """Ignore configuration for the repomix."""

        useGitignore: bool = True  # noqa: N815
        useDefaultPatterns: bool = True  # noqa: N815
        customPatterns: list = []  # noqa: N815

    class Security(BaseModel):
        """Security configuration for the repomix."""

        enableSecurityCheck: bool = True  # noqa: N815

    output: Output = Output()
    include: list = ["**/*"]
    ignore: Ignore = Ignore()
    security: Security = Security()

    def dump_json(self, file_path: str) -> None:
        """Dump the configuration to a JSON file."""
        with Path(file_path).open("w") as file:
            json.dump(self.model_dump(), file, indent=4)


class GroupInfo(BaseModel):
    """Model for group information."""

    group_number: int
    student_1: str | None
    student_2: str | None
    student_3: str | None
    student_4: str | None
    student_5: str | None
    repo_url: str


class TADependency(BaseModel):
    """Model for the dependencies of the TA agent."""

    group_info: GroupInfo
    repomix: RepoMix


class TACodeResponse(BaseModel):
    """Model for the response from the TA agent for the code."""

    code_quality: int = Field(..., ge=1, le=5, description="Score the code quality on a scale from 1 to 5")
    unit_testing: int = Field(
        ..., ge=1, le=5, description="Score the unit testing in the codebase on a scale from 1 to 5"
    )
    ci_cd: int = Field(
        ..., ge=1, le=5, description="Score the continuous integration and deployment process on a scale from 1 to 5"
    )
    summary: str = Field(..., description="Provide a brief summary of the code quality and unit testing")
    overall_score: int = Field(..., ge=1, le=10, description="Overall score from 1-10 for the hole codebase")
    confidence: int = Field(..., ge=1, le=10, description="Confidence in the overall score from 1-10")

    request_usage: Usage | None = None


class TAReportResponse(BaseModel):
    """Model for the response from the TA agent for the report."""

    checklist: int = Field(..., ge=0, le=52, description="How many items from the checklist were completed")
    coding_env: int = Field(..., ge=1, le=5, description="Score for section on coding environment")
    version_control: int = Field(..., ge=1, le=5, description="Score for section on version control")
    code_run_and_experiments: int = Field(..., ge=1, le=5, description="Score for section on code run and experiments")
    cloud: int = Field(..., ge=1, le=5, description="Score for section on cloud")
    deployment: int = Field(..., ge=1, le=5, description="Score for section on deployment")
    overall_discussion: int = Field(..., ge=1, le=5, description="Score for section on overall discussion of project")
    summary: str = Field(..., description="Provide a brief summary of the report evaluation")
    overall_score: int = Field(..., ge=1, le=10, description="Overall score from 1-10 for report")
    confidence: int = Field(..., ge=1, le=10, description="Confidence in the overall score from 1-10")

    request_usage: Usage | None = None
