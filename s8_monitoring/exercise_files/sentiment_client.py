import argparse
import random
import time

import requests

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="http://localhost:8000/predict")
    parser.add_argument("--wait_time", type=int, default=30)
    parser.add_argument("--max_iterations", type=int, default=100)
    args = parser.parse_args()

    reviews = [
        "This app is fantastic! I use it every day.",
        "I enjoy using this app, it's very helpful.",
        "It's a decent app, nothing extraordinary but useful.",
        "This app is okay, but it could be improved.",
        "I'm not very impressed with this app.",
        "This app is not meeting my expectations.",
    ]

    negative_phrases = [
        "It’s getting frustrating to use.",
        "There are so many bugs now.",
        "The app crashes often and I am really disappointed.",
        "I think I'm going to stop using this app soon.",
        "It has become completely unusable.",
    ]

    count = 0
    while count < args.max_iterations:
        review = random.choice(reviews)
        negativity_probability = min(count / args.max_iterations, 1.0)

        updated_review = review
        for phrase in negative_phrases:
            if random.random() < negativity_probability:
                updated_review += " " + phrase

        response = requests.post(args.url, json={"review": updated_review}, timeout=10)
        print(f"Iteration {count}, Sent review: {updated_review}, Response: {response.json()}")
        time.sleep(args.wait_time)
        count += 1
