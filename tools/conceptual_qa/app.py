import datetime

import firebase_admin
import streamlit as st
from firebase_admin import credentials, firestore

st.set_page_config(page_title="Quiz App", layout="centered")


@st.cache_resource
def init_firestore():
    """Initialize Firestore client with application default credentials."""
    if not firebase_admin._apps:  # noqa: SLF001
        cred = credentials.ApplicationDefault()
        firebase_admin.initialize_app(cred)
    return firestore.client(database_id="conceptual")


def get_questions(db: firestore.firestore.Client, day: int) -> list:
    """Fetch quiz questions for the specified day from Firestore."""
    docs = db.collection("quiz_questions").where("day", "==", day).stream()
    questions = []
    for doc in docs:
        data = doc.to_dict()
        data["id"] = doc.id
        questions.append(data)
    return sorted(questions, key=lambda q: q["id"])


def save_response(db: firestore.firestore.Client, student_id: str, answers: dict, score: int, day: int) -> None:
    """Save the student's answers and score to Firestore."""
    collection_name = f"quiz_responses_day{day}"
    db.collection(collection_name).document(student_id).set(
        {
            "student_id": student_id,
            "day": day,
            "answers": answers,
            "score": score,
            "timestamp": datetime.datetime.now(tz=datetime.UTC).isoformat(),
        }
    )


def has_submitted(db: firestore.firestore.Client, student_id: str, day: int) -> bool:
    """Check if the student has already submitted answers for the given day."""
    collection_name = f"quiz_responses_day{day}"
    return db.collection(collection_name).document(student_id).get().exists


def main():  # noqa: C901
    """Main function to run the quiz app."""
    # Get quiz day from URL query params
    params = st.query_params
    day = int(params.get("day", [None])[0]) if "day" in params else None

    if day is None or not (1 <= day <= 10):
        st.error("No valid quiz day specified. Add `?day=1` (or another day number) to the URL.")
        st.stop()

    st.title(f"Multiple-Choice Quiz – Day {day}")
    db = init_firestore()
    show_answers = True  # Show correct answers after submission

    # Student ID input
    if "student_id" not in st.session_state:
        st.session_state["student_id"] = ""

    st.session_state["student_id"] = st.text_input(
        "Enter your student ID (e.g., S123456)", value=st.session_state["student_id"]
    )

    if not st.session_state["student_id"]:
        st.stop()

    student_id = st.session_state["student_id"].strip().upper()
    if not (student_id.startswith("S") and student_id[1:].isdigit() and len(student_id) == 7):
        st.warning("Invalid student ID format.")
        st.stop()

    if has_submitted(db, student_id, day):
        st.info("✅ You already submitted answers for this day.")
        st.stop()

    # Load and display questions
    questions = get_questions(db, day)
    if not questions:
        st.warning("No questions found for this day.")
        st.stop()

    st.markdown("---")
    st.subheader("Quiz")

    answers = {}
    for q in questions:
        st.markdown(f"**{q['id']}. {q['question']}**")
        options = list(q["options"].items())
        option_texts = [f"({k}) {v}" for k, v in options]
        selected = st.radio(f"Your answer for {q['id']}", option_texts, key=q["id"])
        selected_key = selected[1]  # e.g., (b) text → extract "b"
        answers[q["id"]] = selected_key

    if st.button("Submit Answers"):
        score = sum(answers[q["id"]] == q["correct"] for q in questions)
        save_response(db, student_id, answers, score, day)
        st.success(f"🎉 Submission successful! Your score: {score}/{len(questions)}")

        if show_answers:
            st.markdown("---")
            st.subheader("Correct Answers")
            for q in questions:
                correct_key = q["correct"]
                correct_ans = f"({correct_key}) {q['options'][correct_key]}"
                user_key = answers[q["id"]]
                user_ans = f"({user_key}) {q['options'][user_key]}"
                st.markdown(
                    f"**{q['id']}**: {'✅' if user_key == correct_key else '❌'}  \n"
                    f"Your answer: {user_ans}  \n"
                    f"Correct answer: {correct_ans}"
                )


if __name__ == "__main__":
    main()
