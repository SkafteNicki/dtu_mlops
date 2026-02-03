import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase app
cred = credentials.ApplicationDefault()
firebase_admin.initialize_app(cred)
db = firestore.client(database_id="conceptual")

# Example questions to upload
day1 = [
    {
        "id": "Q1",
        "question": "What is the primary reason for integrating Continuous Integration into a software dev workflow?",
        "options": {
            "a": "To automate code formatting and linting",
            "b": "To ensure that new changes do not break existing functionality through automated tests",
            "c": "To deploy code automatically to users after every commit",
            "d": "To eliminate the need for manual code reviews",
            "e": "To improve the performance of the application in production",
        },
        "correct": "b",
        "explanation": "CI is about early detection of issues by running tests automatically when code changes.",
    },
    {
        "id": "Q2",
        "question": "Which of the following is NOT a benefit of using Continuous Integration in software development?",
        "options": {
            "a": "Faster feedback on code changes",
            "b": "Reduced integration problems",
            "c": "Increased manual testing efforts",
            "d": "Improved collaboration among team members",
            "e": "Early detection of bugs",
        },
        "correct": "c",
        "explanation": "CI aims to reduce manual testing by automating the testing process.",
    },
    {
        "id": "Q3",
        "question": "What is the role of automated tests in Continuous Integration (CI)?",
        "options": {
            "a": "To replace manual testing entirely",
            "b": "To ensure that new code changes do not introduce new bugs or break existing functionality",
            "c": "To provide detailed documentation for the codebase",
            "d": "To improve the performance of the application in production",
            "e": "To automate the deployment process",
        },
        "correct": "b",
        "explanation": "Automated tests are crucial in CI to catch issues early in the development cycle.",
    },
]

# Upload questions to Firestore
days_questions = {
    1: day1,
    # 2: day2,
    # add more days as needed
}

for day, questions in days_questions.items():
    for q in questions:
        doc_ref = db.collection("quiz_questions").document(q["id"])
        doc_ref.set(
            {
                "question": q["question"],
                "options": q["options"],
                "correct": q["correct"],
                "explanation": q.get("explanation", ""),
                "day": day,
            }
        )
    print(f"Uploaded questions for Day {day}")
