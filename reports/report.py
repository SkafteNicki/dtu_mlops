# run following commands to install requirements
# pip install click
# pip install markdown

import re
import warnings
from functools import partial

import click
import markdown


class TeacherWarning(UserWarning):
    """Warning raised when a teacher check fails."""

    pass


@click.group()
def cli():
    """CLI for report."""
    pass


@cli.command()
def html():
    """Convert README.md to html page."""
    with open("README.md", "r") as file:
        text = file.read()
    text = text[43:]  # remove header

    html = markdown.markdown(text)

    with open("report.html", "w") as newfile:
        newfile.write(html)


@cli.command()
def check():
    """Check if report satisfies the requirements."""
    with open("README.md", "r") as file:
        text = file.read()
    text = text[43:]  # remove header

    answers = []
    per_question = text.split("Answer:")
    for q in per_question:
        if "###" in q:
            q = q.split("###")[0]
            if "##" in q:
                q = q.split("##")[0]
            answers.append(q)

    answers.append(per_question[-1])
    answers = answers[1:]  # remove first section
    answers = [answer.strip("\n") for answer in answers]

    def no_constraints(answer, index):
        pass

    def length_constraints(answer, index, min_length, max_length):
        answer = answer.split()
        if not (min_length <= len(answer) <= max_length):
            warnings.warn(
                f"Question {index} failed check. Expected number of words to be"
                f" between {min_length} and {max_length} but got {len(answer)}",
                TeacherWarning,
            )

    def image_constrains(answer, index, min_length, max_length):
        links = re.findall(r"\!\[.*?\]\(.*?\)", answer)
        if not (min_length <= len(links) <= max_length):
            warnings.warn(
                f"Question {index} failed check. Expected number of screenshots to be"
                f" between {min_length} and {max_length} but got {len(links)}",
                TeacherWarning,
            )

    def multi_constrains(answer, index, constrains):
        for fn in constrains:
            fn(answer, index)

    question_constrains = [
        no_constraints,
        no_constraints,
        partial(length_constraints, min_length=100, max_length=200),
        partial(length_constraints, min_length=100, max_length=200),
        partial(length_constraints, min_length=100, max_length=200),
        partial(length_constraints, min_length=50, max_length=100),
        partial(length_constraints, min_length=50, max_length=100),
        partial(length_constraints, min_length=100, max_length=200),
        partial(length_constraints, min_length=100, max_length=200),
        partial(length_constraints, min_length=100, max_length=200),
        partial(length_constraints, min_length=200, max_length=300),
        partial(length_constraints, min_length=50, max_length=100),
        partial(length_constraints, min_length=100, max_length=200),
        partial(
            multi_constrains,
            constrains=(
                partial(length_constraints, min_length=200, max_length=300),
                partial(image_constrains, min_length=1, max_length=3),
            ),
        ),
        partial(length_constraints, min_length=100, max_length=200),
        partial(length_constraints, min_length=100, max_length=200),
        partial(length_constraints, min_length=50, max_length=200),
        partial(length_constraints, min_length=100, max_length=200),
        partial(image_constrains, min_length=1, max_length=2),
        partial(image_constrains, min_length=1, max_length=1),
        partial(image_constrains, min_length=1, max_length=1),
        partial(length_constraints, min_length=100, max_length=200),
        partial(length_constraints, min_length=100, max_length=200),
        partial(length_constraints, min_length=25, max_length=100),
        partial(
            multi_constrains,
            constrains=(
                partial(length_constraints, min_length=200, max_length=400),
                partial(image_constrains, min_length=1, max_length=1),
            ),
        ),
        partial(length_constraints, min_length=200, max_length=400),
        partial(length_constraints, min_length=50, max_length=200),
    ]
    if len(answers) != 27:
        raise ValueError("Number of answers are different from the expected 27. Have you filled out every field?")

    for i, (answer, const) in enumerate(zip(answers, question_constrains), start=1):
        const(answer, i)


if __name__ == "__main__":
    cli()
