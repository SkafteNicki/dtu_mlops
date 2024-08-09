import os

from invoke import task

CURRENT_DIR = os.getcwd()
WINDOWS = os.name == "nt"


@task
def install(ctx):
    """Create the environment for course."""
    ctx.run("conda create -n dtumlops python=3.11 pip --no-default-packages -y", echo=True, pty=not WINDOWS)
    ctx.run("conda activate dtumlops", echo=True, pty=not WINDOWS)
    ctx.run("pip install -r requirements.txt", echo=True, pty=not WINDOWS)


@task
def precommit(ctx):
    """Install and run pre-commit checks."""
    ctx.run("pre-commit install", echo=True, pty=not WINDOWS)
    ctx.run("pre-commit run --all-files", echo=True, pty=not WINDOWS)


@task(aliases=["mkdocs"])
def docs(ctx):
    """Build the documentation."""
    ctx.run("mkdocs serve --dirty", echo=True, pty=not WINDOWS)


@task
def lint(ctx):
    """Run linters."""
    ctx.run("ruff check . --fix", echo=True, pty=not WINDOWS, warn=True)
    ctx.run("ruff format .", echo=True, pty=not WINDOWS, warn=True)
    ctx.run("mypy .", echo=True, pty=not WINDOWS)


@task
def linkcheck(ctx):
    """Check for broken links."""
    ctx.run(
        (
            "docker build "
            "https://github.com/gaurav-nelson/github-action-markdown-link-check.git#master -t linkcheck:latest"
        ),
        echo=True,
        pty=not WINDOWS,
    )
    ctx.run(
        (
            "docker run "
            "--rm "
            f"-v {CURRENT_DIR}:/github/workspace "
            "linkcheck "
            """no no /github/workspace/.github/linkcheck_config.json /github/workspace -1 no master .md " " """
        ),
        echo=True,
        pty=not WINDOWS,
    )


@task(aliases=["drawio"])
def diagrams(ctx):
    """Generate diagrams."""
    ctx.run("docker pull rlespinasse/drawio-export", echo=True, pty=not WINDOWS)
    ctx.run(
        (
            f"docker run --rm -v {CURRENT_DIR}:/data rlespinasse/drawio-export "
            "figures/diagrams/ -f png --on-changes --remove-page-suffix --output ../"
        ),
        echo=True,
        pty=not WINDOWS,
    )
    ctx.run(
        (
            f"docker run --rm -v {CURRENT_DIR}:/data rlespinasse/drawio-export "
            "figures/diagrams/mlops_canvas.drawio -f pdf --on-changes --remove-page-suffix --output ../.."
        ),
        echo=True,
        pty=not WINDOWS,
    )
