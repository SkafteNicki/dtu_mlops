# run following commands to install requirements
# pip install click
# pip install markdown

import click
import markdown
import warnings
import re
from functools import partial


class TeacherWarning(UserWarning):
    pass


@click.group()
def cli():
    pass


@cli.command()
def html():
    with open("README.md", "r") as file:
        text = file.read()
    text = text[43:]  # remove header

    html = markdown.markdown(text)

    with open("report.html", "w") as newfile:
        newfile.write(html)


@cli.command()
def check():
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
    answers = [ans.strip("\n") for ans in answers]

    def no_constraints(answer, index):
        pass

    def length_constraints(answer, index, min, max):
        answer = answer.split()
        if not (min <= len(answer) <= max):
            warnings.warn(
                f"Question {index} failed check. Expected number of words to be"
                f" between {min} and {max} but got {len(answer)}",
                TeacherWarning,
            )

    def image_constrains(answer, index, min, max):
        links = re.findall(r"\!\[.*?\]\(.*?\)", answer)
        if not (min <= len(links) <= max):
            warnings.warn(
                f"Question {index} failed check. Expected number of screenshots to be"
                f" between {min} and {max} but got {len(links)}",
                TeacherWarning,
            )

    def multi_constrains(answer, index, constrains):
        for fn in constrains:
            fn(answer, index)

    question_constrains = [
        no_constraints,
        no_constraints,
        partial(length_constraints, min=100, max=200),
        partial(length_constraints, min=100, max=200),
        partial(length_constraints, min=100, max=200),
        partial(length_constraints, min=50, max=100),
        partial(length_constraints, min=50, max=100),
        partial(length_constraints, min=100, max=200),
        partial(length_constraints, min=100, max=200),
        partial(length_constraints, min=100, max=200),
        partial(length_constraints, min=200, max=300),
        partial(length_constraints, min=50, max=100),
        partial(length_constraints, min=100, max=200),
        partial(
            multi_constrains,
            constrains=(
                partial(length_constraints, min=200, max=300),
                partial(image_constrains, min=1, max=3),
            ),
        ),
        partial(length_constraints, min=100, max=200),
        partial(length_constraints, min=100, max=200),
        partial(length_constraints, min=50, max=200),
        partial(length_constraints, min=100, max=200),
        partial(image_constrains, min=1, max=2),
        partial(image_constrains, min=1, max=1),
        partial(image_constrains, min=1, max=1),
        partial(length_constraints, min=100, max=200),
        partial(length_constraints, min=100, max=200),
        partial(length_constraints, min=25, max=100),
        partial(
            multi_constrains,
            constrains=(
                partial(length_constraints, min=200, max=400),
                partial(image_constrains, min=1, max=1),
            ),
        ),
        partial(length_constraints, min=200, max=400),
        partial(length_constraints, min=50, max=200),
    ]
    if len(answers) != 27:
        raise ValueError(
            "Number of answers are different from the expected 27. Have you filled out every field?"
        )

    for i, (ans, const) in enumerate(zip(answers, question_constrains)):
        const(ans, i)


if __name__ == "__main__":
    cli()
