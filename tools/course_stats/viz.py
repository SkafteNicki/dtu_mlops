import json
import os

import matplotlib.pyplot as plt
import numpy as np
import typer
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI
from playwright.sync_api import sync_playwright
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.usage import Usage
from wordcloud import STOPWORDS, WordCloud

load_dotenv()
plt.rcParams.update({"font.size": 16})

os.makedirs("output", exist_ok=True)

app = typer.Typer()


@app.command()
def plot_students():
    """Plot the number of students over the years."""
    year = [2021, 2022, 2023, 2024, 2025, 2026]
    students = [60, 102, 196, 275, 374, 450]

    # Calculate year-over-year percentage increase
    percentage_increase = [0]  # No increase for the first year
    for i in range(1, len(students)):
        increase = ((students[i] - students[i - 1]) / students[i - 1]) * 100
        percentage_increase.append(increase)

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(year, students, marker="o", linestyle="-", color="b", linewidth=2)

    # Add text for percentage increase
    for i in range(1, len(year)):
        plt.text(0.99995 * year[i], students[i], f"{percentage_increase[i]:.2f}%", fontsize=16, ha="right")

    plt.xlabel("Year")
    plt.ylabel("Number of Students")
    plt.title("Year-over-Year Increase in Number of Students")
    plt.grid(True)

    # Set x-axis ticks to be integers
    plt.xticks(year)

    plt.savefig("output/students.png", dpi=300, bbox_inches="tight")


@app.command()
def evaluations():
    """Plot the number of students and responses per year and the mean score for each question."""
    data2022 = [
        (102, 29, 0, 28),
        ("Yes", 0, 0, 0, 1, 28, 5),
        ("Yes", 0, 0, 0, 3, 26, 4.9),
        ("Yes", 0, 1, 1, 8, 19, 4.6),
        ("Yes", 0, 3, 11, 8, 7, 3.7),
        ("Yes", 0, 1, 2, 16, 10, 4.2),
        ("Yes", 0, 2, 9, 16, 2, 3.6),
    ]
    data2022 = [d for data in data2022 for d in data]
    data2023 = [
        (196, 30, 0, 15),
        ("Yes", 0, 0, 1, 5, 24, 4.8),
        ("Yes", 0, 0, 0, 4, 26, 4.9),
        ("Yes", 0, 1, 4, 6, 19, 4.4),
        ("Yes", 2, 4, 9, 6, 9, 3.5),
        ("Yes", 0, 1, 2, 8, 19, 4.5),
        ("Yes", 0, 2, 13, 5, 10, 3.8),
    ]
    data2023 = [d for data in data2023 for d in data]
    data2024 = [
        (274, 45, 1, 16),
        ("Yes", 0, 0, 2, 5, 38, 4.8),
        ("Yes", 0, 1, 1, 6, 36, 4.8),
        ("Yes", 0, 1, 2, 8, 33, 4.7),
        ("Yes", 3, 4, 14, 10, 13, 3.6),
        ("Yes", 1, 2, 2, 13, 26, 4.4),
        ("Yes", 1, 4, 18, 13, 8, 3.5),
    ]
    data2024 = [d for data in data2024 for d in data]
    data2025 = [
        (374, 50, 0, 13),
        ("Yes", 0, 0, 2, 5, 43, 4.8),
        ("Yes", 0, 0, 1, 5, 43, 4.9),
        ("Yes", 0, 1, 3, 16, 28, 4.5),
        ("Yes", 2, 3, 17, 15, 12, 3.7),
        ("Yes", 0, 1, 6, 12, 30, 4.4),
        ("Yes", 1, 4, 24, 15, 4, 3.4),
    ]
    data2025 = [d for data in data2025 for d in data]
    data = [data2022, data2023, data2024, data2025]

    years = [2022, 2023, 2024, 2025]
    number_of_students = [d[0] for d in data]
    number_of_responses = [d[1] for d in data]

    # Calculate the percentage of responses
    percentage_responses = [
        (responses / students) * 100 for responses, students in zip(number_of_responses, number_of_students)
    ]

    # Plot the data
    bar_width = 0.35
    index = np.arange(len(years))

    plt.figure(figsize=(10, 6))
    plt.bar(index, number_of_students, bar_width, label="Number of Students")
    plt.bar(index + bar_width, number_of_responses, bar_width, label="Number of Responses")

    # Add text for percentage responses
    for i in range(len(years)):
        plt.text(
            index[i] + bar_width / 2,
            number_of_responses[i],
            f"{percentage_responses[i]:.2f}%",
            ha="center",
            va="bottom",
        )

    plt.xlabel("Year")
    plt.ylabel("Count")
    plt.title("Number of Students and Responses per Year")
    plt.xticks(index + bar_width / 2, years)
    plt.legend()
    plt.grid(True)
    plt.savefig("output/students_responses.png", dpi=300, bbox_inches="tight")
    plt.clf()

    scores = [1, 2, 3, 4, 5]
    q1_score_count = [d[5:10] for d in data]
    q2_score_count = [d[12:17] for d in data]
    q3_score_count = [d[19:24] for d in data]
    q4_score_count = [d[26:31] for d in data]
    q5_score_count = [d[33:38] for d in data]
    q6_score_count = [d[40:45] for d in data]

    q1_score_mean = [np.average(scores, weights=score_count) for score_count in q1_score_count]
    q2_score_mean = [np.average(scores, weights=score_count) for score_count in q2_score_count]
    q3_score_mean = [np.average(scores, weights=score_count) for score_count in q3_score_count]
    q4_score_mean = [np.average(scores, weights=score_count) for score_count in q4_score_count]
    q5_score_mean = [np.average(scores, weights=score_count) for score_count in q5_score_count]
    q6_score_mean = [np.average(scores, weights=score_count) for score_count in q6_score_count]
    q_means = [q1_score_mean, q2_score_mean, q3_score_mean, q4_score_mean, q5_score_mean, q6_score_mean]

    def weighted_stdev(values, weights):
        average = np.average(values, weights=weights)
        variance = np.average((values - average) ** 2, weights=weights)
        return np.sqrt(variance)

    q1_score_stdev = [weighted_stdev(scores, score_count) for score_count in q1_score_count]
    q2_score_stdev = [weighted_stdev(scores, score_count) for score_count in q2_score_count]
    q3_score_stdev = [weighted_stdev(scores, score_count) for score_count in q3_score_count]
    q4_score_stdev = [weighted_stdev(scores, score_count) for score_count in q4_score_count]
    q5_score_stdev = [weighted_stdev(scores, score_count) for score_count in q5_score_count]
    q6_score_stdev = [weighted_stdev(scores, score_count) for score_count in q6_score_count]
    q_stdev = [q1_score_stdev, q2_score_stdev, q3_score_stdev, q4_score_stdev, q5_score_stdev, q6_score_stdev]

    questions = [
        "I have learned a lot from this course.",
        "The learning activities of the course were in line with the learning objectives of the course.",
        "The learning activities motivated me to work with the material.",
        "During the course, I have had the opportunity to get feedback on my performance.",
        "It was generally clear what was expected of me in exercises, project work, etc.",
        """5 ECTS credits correspond 45 working hours per week for the three-week period.
        I think the time I have spent on this course is""",
    ]

    for question_index, question in enumerate(questions):
        plt.bar(years, q_means[question_index], yerr=q_stdev[question_index], capsize=5)
        plt.xticks(years)
        plt.xlabel("Year")
        plt.ylabel("Mean Score")
        # Divide the title into multiple lines if it is too long
        max_title_length = 50
        if len(question) > max_title_length:
            words = question.split()
            question_lines = []
            current_line = ""
            for word in words:
                if len(current_line) + len(word) + 1 <= max_title_length:
                    current_line += word + " "
                else:
                    question_lines.append(current_line.strip())
                    current_line = word + " "
            question_lines.append(current_line.strip())
            question = "\n".join(question_lines)
        plt.title(question)
        plt.grid(True)
        plt.savefig(f"output/question{question_index + 1}.png", dpi=300, bbox_inches="tight")
        plt.clf()


@app.command()
def extract_course_evaluations():
    """Extract course evaluations from DTU's evaluation system."""
    USER = os.getenv("USER")  # noqa: N806
    PASSWORD = os.getenv("PASSWORD")  # noqa: N806

    courses = [
        "Machine Learning Operations Jan 22",
        "Machine Learning Operations Jan 23",
        "Machine Learning Operations Jan 24",
        "Machine Learning Operations Jan 25",
    ]

    course_data = []
    with sync_playwright() as p:
        browser = p.firefox.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        page.goto("https://evaluering.dtu.dk")
        page.get_by_placeholder("User").fill(USER)
        page.get_by_placeholder("Password").fill(PASSWORD)
        page.get_by_role("button", name="Sign in").click()

        for course in courses:
            page.get_by_role("link", name="Show more").click()
            page.get_by_role("link", name=course).click()
            course_content = page.locator("article").filter(has_text="3.1 Here you can write").text_content()
            course_data.append(course_content)
            page.get_by_role("link", name="My Evaluations").click()

        context.close()
        browser.close()

    client = AsyncAzureOpenAI(
        azure_endpoint=os.environ.get("AZURE_ENDPOINT"),
        api_version=os.environ.get("API_VERSION"),
        api_key=os.environ.get("API_KEY"),
    )
    model = OpenAIModel(os.environ.get("MODEL"), openai_client=client)

    class ResultType(BaseModel):
        result: list[str]
        summary: str
        sentiment: bool

    agent = Agent(
        model=model,
        system_prompt="""
        Your are an assistant for a university that needs to help the course responsible format the evaluations
        from the students correct. You will be receiving a bunch of evaluation, some will be in danish, some in english.
        Please return the evaluations in a structured format, as a list of strings, a summary of the evaluations
        and a sentiment if the evaluations are positive or negative. All danish evaluations should be translated
        to english.
        """,
        result_type=ResultType,
    )

    class SaveType(BaseModel):
        data: ResultType
        usage: Usage
        year: int

    results: list[SaveType] = []
    for data in course_data:
        result = agent.run_sync(data)
        results.append(SaveType(data=result.data, usage=result.usage(), year=2022 + len(results)))

    with open("output/evaluations.json", "w") as f:
        json.dump([result.model_dump() for result in results], f, indent=4)


@app.command()
def wordcloud(all_years_together: bool = False):
    """Construct a word cloud from the course evaluations."""
    if "output/evaluations.json" not in os.listdir("output"):
        extract_course_evaluations()
    with open("output/evaluations.json") as f:
        data = json.load(f)
    evaluations = []
    for d in data:
        evaluations.append(d["data"]["result"])

    custom_stopwords = set(STOPWORDS)
    custom_stopwords.update(["course", "courses"])

    if all_years_together:
        evaluations = [e for evals in evaluations for e in evals]
        text = " ".join(evaluations)

        wordcloud = WordCloud(width=800, height=400, stopwords=custom_stopwords).generate(text)
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.savefig("output/wordcloud.png", dpi=300, bbox_inches="tight")
    else:
        for i, evals in enumerate(evaluations):
            text = " ".join(evals)
            wordcloud = WordCloud(width=800, height=400, stopwords=custom_stopwords).generate(text)
            plt.figure(figsize=(10, 6))
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            plt.savefig(f"output/wordcloud_{2022 + i}.png", dpi=300, bbox_inches="tight")


@app.command()
def extract_evaluations():
    """Extract course evaluations from DTU's evaluation system."""
    USER = os.getenv("USER")  # noqa: N806
    PASSWORD = os.getenv("PASSWORD")  # noqa: N806

    courses = [
        "Machine Learning Operations Jan 22",
        "Machine Learning Operations Jan 23",
        "Machine Learning Operations Jan 24",
        "Machine Learning Operations Jan 25",
    ]

    class CourseData(BaseModel):
        course: str
        course_feedback: str
        course_download: str
        exam_feedback: str
        exam_download: str
        teacher_feedback: str
        teacher_download: str

    course_data: list[CourseData] = []
    with sync_playwright() as p:
        browser = p.firefox.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        page.goto("https://evaluering.dtu.dk")
        page.get_by_placeholder("User").fill(USER)
        page.get_by_placeholder("Password").fill(PASSWORD)
        page.get_by_role("button", name="Sign in").click()

        for course in courses:
            page.get_by_role("link", name="Show more").click()
            page.get_by_role("link", name=course).click()

            course_feedback = page.locator("article").filter(has_text="3.1 Here you can write").text_content()
            with page.expect_download() as download_info:
                page.get_by_role("button", name="Download Exceldata").click()
            course_download = download_info.value
            course_download.save_as(os.path.join("data", f"{course}_course_feedback.xlsx"))

            page.get_by_role("link", name="Evaluation of exam").click()
            exam_feedback = page.locator("article").filter(has_text="6 Further comments /").text_content()
            with page.expect_download() as download1_info:
                page.get_by_role("button", name="Download Exceldata").click()
            exam_download = download1_info.value
            exam_download.save_as(os.path.join("data", f"{course}_exam_feedback.xlsx"))

            page.get_by_role("link", name="Schema B1, Teacher").click()
            teacher_feedback = page.locator("article").filter(has_text="2.1 Do you have constructive").text_content()
            with page.expect_download() as download2_info:
                page.get_by_role("button", name="Download Exceldata").click()
            teacher_download = download2_info.value
            teacher_download.save_as(os.path.join("data", f"{course}_teacher_feedback.xlsx"))

            course_data.append(
                CourseData(
                    course=course,
                    course_feedback=course_feedback,
                    course_download=course_download.suggested_filename,
                    exam_feedback=exam_feedback,
                    exam_download=exam_download.suggested_filename,
                    teacher_feedback=teacher_feedback,
                    teacher_download=teacher_download.suggested_filename,
                )
            )

            page.get_by_role("link", name="My Evaluations").click()  # reset to homepage

        context.close()
        browser.close()


if __name__ == "__main__":
    app()
