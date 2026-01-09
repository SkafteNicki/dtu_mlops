# run following commands to install requirements
# pip install typer markdown pydantic loguru
# or
# uv add typer markdown pydantic loguru

import re
from pathlib import Path

import markdown
import pydantic
import typer
from loguru import logger


class Constraints(pydantic.BaseModel):
    """Base class for constraints."""

    def __call__(self, answer: str, index: int) -> None:
        """Check constraints on the answer."""
        raise NotImplementedError


class NoConstraints(Constraints):
    """No constraints on the answer."""

    def __call__(self, answer: str, index: int) -> bool:
        """No constraints on the answer."""
        return True


class LengthConstraints(Constraints):
    """Check constraints on the length of the answer."""

    min_length: int = pydantic.Field(ge=0)
    max_length: int = pydantic.Field(ge=0)

    def __call__(self, answer: str, index: int) -> bool:
        """Check constraints on the length of the answer."""
        answer = answer.split()
        if not (self.min_length <= len(answer) <= self.max_length):
            logger.warning(
                f"Question {index} failed check. Expected number of words to be"
                f" between {self.min_length} and {self.max_length} but got {len(answer)}"
            )
            return False
        return True


class ImageConstraints(Constraints):
    """Check constraints on the number of images in the answer."""

    min_images: int = pydantic.Field(ge=0)
    max_images: int = pydantic.Field(ge=0)

    def __call__(self, answer: str, index: int) -> bool:
        """Check constraints on the number of images in the answer."""
        links = re.findall(r"\!\[.*?\]\(.*?\)", answer)
        if not (self.min_images <= len(links) <= self.max_images):
            logger.warning(
                f"Question {index} failed check. Expected number of screenshots to be"
                f" between {self.min_images} and {self.max_images} but got {len(links)}"
            )
            return False
        return True


class MultiConstraints(Constraints):
    """Check multiple constraints on the answer."""

    constrains: list[Constraints]

    def __call__(self, answer: str, index: int) -> None:
        """Check multiple constraints on the answer."""
        value = True
        for fn in self.constrains:
            value = fn(answer, index) and value
        return value


app = typer.Typer()


@app.command()
def html() -> None:
    """Convert README.md to html page."""
    with Path("README.md").open() as file:
        text = file.read()
    text = text[43:]  # remove header

    html = markdown.markdown(text)

    with open("report.html", "w") as newfile:
        newfile.write(html)


@app.command()
def check() -> None:
    """Check if report satisfies the requirements."""
    with Path("README.md").open() as file:
        text = file.read()

    # answers in general can be found between "Answer:" and "###" or "##"
    # which marks the next question or next section
    answers = []
    per_question = text.split("Answer:")
    per_question.pop(0)  # remove the initial section
    for q in per_question:
        if "###" in q:
            q = q.split("###")[0]
            if "##" in q:
                q = q.split("##")[0]
            answers.append(q)

    # add the last question
    answers.append(per_question[-1])

    # remove newlines
    answers = [answer.strip("\n") for answer in answers]

    question_constraints = {
        "question_1": NoConstraints(),
        "question_2": NoConstraints(),
        "question_3": LengthConstraints(min_length=0, max_length=200),
        "question_4": LengthConstraints(min_length=100, max_length=200),
        "question_5": LengthConstraints(min_length=100, max_length=200),
        "question_6": LengthConstraints(min_length=100, max_length=200),
        "question_7": LengthConstraints(min_length=50, max_length=100),
        "question_8": LengthConstraints(min_length=100, max_length=200),
        "question_9": LengthConstraints(min_length=100, max_length=200),
        "question_10": LengthConstraints(min_length=100, max_length=200),
        "question_11": LengthConstraints(min_length=200, max_length=300),
        "question_12": LengthConstraints(min_length=50, max_length=100),
        "question_13": LengthConstraints(min_length=100, max_length=200),
        "question_14": MultiConstraints(
            constrains=[
                LengthConstraints(min_length=200, max_length=300),
                ImageConstraints(min_images=1, max_images=3),
            ]
        ),
        "question_15": LengthConstraints(min_length=100, max_length=200),
        "question_16": LengthConstraints(min_length=100, max_length=200),
        "question_17": LengthConstraints(min_length=50, max_length=200),
        "question_18": LengthConstraints(min_length=100, max_length=200),
        "question_19": ImageConstraints(min_images=1, max_images=2),
        "question_20": ImageConstraints(min_images=1, max_images=2),
        "question_21": ImageConstraints(min_images=1, max_images=2),
        "question_22": LengthConstraints(min_length=100, max_length=200),
        "question_23": LengthConstraints(min_length=100, max_length=200),
        "question_24": LengthConstraints(min_length=100, max_length=200),
        "question_25": LengthConstraints(min_length=100, max_length=200),
        "question_26": LengthConstraints(min_length=100, max_length=200),
        "question_27": LengthConstraints(min_length=100, max_length=200),
        "question_28": LengthConstraints(min_length=0, max_length=200),
        "question_29": MultiConstraints(
            constrains=[
                LengthConstraints(min_length=200, max_length=400),
                ImageConstraints(min_images=1, max_images=1),
            ]
        ),
        "question_30": LengthConstraints(min_length=200, max_length=400),
        "question_31": LengthConstraints(min_length=50, max_length=300),
    }
    if len(answers) != 31:
        msg = "Number of answers are different from the expected 31. Have you changed the template?"
        raise ValueError(msg)

    counter = 0
    for index, (answer, (_, constraints)) in enumerate(zip(answers, question_constraints.items()), 1):
        counter += int(constraints(answer, index))
    logger.info(f"Total number of questions passed: {counter}/{len(answers)}")


if __name__ == "__main__":
    app()
