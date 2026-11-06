import os
import sys
from pathlib import Path

from invoke import task

CURRENT_DIR = os.getcwd()
WINDOWS = os.name == "nt"


def _require_external_venv() -> None:
    """Exit with instructions, as Zensical 0.1.0 hangs when building this course with a real `.venv` in the repo."""
    venv = Path(CURRENT_DIR, ".venv")
    if venv.is_dir() and not venv.is_symlink():
        sys.exit(
            "Zensical hangs with a .venv in the repo root. Remove it, run "
            "`export UV_PROJECT_ENVIRONMENT=../dtu_mlops-venv`, then run this task again."
        )


@task
def install(ctx) -> None:
    """Create the environment for course."""
    ctx.run("uv sync --all-groups", echo=True, pty=not WINDOWS)


@task
def precommit(ctx) -> None:
    """Install and run pre-commit checks."""
    ctx.run("uv run pre-commit install", echo=True, pty=not WINDOWS)
    ctx.run("uv run pre-commit run --all-files", echo=True, pty=not WINDOWS)


@task(aliases=["mkdocs"])
def docs(ctx) -> None:
    """Serve the documentation locally."""
    _require_external_venv()
    ctx.run("uv run zensical serve", echo=True, pty=not WINDOWS)


@task
def lint(ctx) -> None:
    """Run linters."""
    ctx.run("uv run ruff check . --fix", echo=True, pty=not WINDOWS, warn=True)
    ctx.run("uv run ruff format .", echo=True, pty=not WINDOWS, warn=True)
    ctx.run("uv run mypy .", echo=True, pty=not WINDOWS)


@task
def docker_running(ctx):
    """Check if Docker is running."""
    result = ctx.run("docker info", pty=not WINDOWS, hide=True, warn=True)

    if result.ok:
        print("Docker is running.")
    else:
        print("Docker is not running.")
        sys.exit(1)


@task(docker_running)
def linkcheck(ctx, offline: bool = False) -> None:
    """Check for broken links. Use --offline to only check local links and anchors."""
    args = "--offline --include-fragments " if offline else ""
    ctx.run(
        f'docker run --rm -v "{CURRENT_DIR}":/input:ro -w /input lycheeverse/lychee:0.24.2 {args}"./**/*.md"',
        echo=True,
        pty=not WINDOWS,
    )


@task
def deploy(ctx, version: str = "latest") -> None:
    """Deploy latest version locally (for testing)."""
    _require_external_venv()
    if version == "latest":
        ctx.run("uv run mike deploy --update-aliases latest stable", echo=True, pty=not WINDOWS)
        ctx.run("uv run mike set-default latest", echo=True, pty=not WINDOWS)
    else:
        ctx.run("uv run mike deploy dev experimental", echo=True, pty=not WINDOWS)


@task
def serve_versioned(ctx) -> None:
    """Serve versioned docs locally."""
    ctx.run("uv run mike serve", echo=True, pty=not WINDOWS)


@task
def list_versions(ctx) -> None:
    """List all deployed versions."""
    ctx.run("uv run mike list", echo=True, pty=not WINDOWS)
