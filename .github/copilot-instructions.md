> General instructions and guidelines for automatic code agents like Github Copilot for interpreting
> and generating code in this repository.

This repository contains course material for the Machine Learning Operations (MLOps) course at the Technical University
of Denmark (DTU). The audience for this repository is therefore primarily computer science students that have some prior
experience with programming in Python and machine learning, and is looking to learn the basics of MLOps. Below is a
brief overview of course content as of right now:

Proper coding environments, code organization, good coding practices, code and data version control, reproducible and
containerized environments, reproducible experiment management, debugging tools, code profiling, large scale
collaborative experiment logging and monitoring, unit testing, continuous integration, continuous machine learning,
cloud infrastructure, cloud based machine learning, distributed data loading and training, optimization methods for
inference, local and cloud based deployment, monitoring of deployed applications.

## Project structure

The course is setup as a mkdocs project. Importantly, the structure is flattened such that all course material is in the
root of the repository. This means that:

- The course is divided into several sessions, named `s1_*`, `s2_*`, etc. Each session folder contains multiple `*.md`
    files which are different learning modules within that session. In addition the `pages/` folder contains additional
    pages for the course website.

- For each session folder there is a subdir called `exercise_files/` which contains the relevant exercise files (e.g.
    Jupyter Notebooks, Python scripts, data files, etc.) for all modules in that session.

- Figures for all pages are stored in the `figures/` folder.

- An additional folder `tools/` contains various utility scripts used for course management and should not be modified
    unless explicitly stated.

## Development environment

- The project uses `uv` for dependency management, meaning that you should use `uv run` to execute commands
    - instead of `python script.py`, use `uv run script.py`
    - instead of `pip install package`, use `uv add package`.
        - If the dependency is is only needed for development, use `uv add --dev package`
        - If the dependency is only needed for optional features, use `uv add --group extra package`
    - instead of `command ...` use `uv run command ...`
- Additionally the project uses `invoke` for task management, so you can run tasks defined in `tasks.py` using
    `uv run invoke <task_name>`. Run `uv run invoke --list` to see all available tasks.
- Project uses `pre-commit` for git hooks. Make sure to run the pre-commit hooks after making changes to the codebase:
    `uv run pre-commit`
