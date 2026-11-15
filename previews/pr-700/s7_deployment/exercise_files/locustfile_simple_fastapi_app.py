import random

from locust import HttpUser, between, task


class MyUser(HttpUser):
    """A simple Locust user class that defines the tasks to be performed by the users."""

    wait_time = between(1, 2)

    @task
    def get_root(self) -> None:
        """A task that simulates a user visiting the root URL of the FastAPI app."""
        self.client.get("/")

    @task(3)
    def get_item(self) -> None:
        """A task that simulates a user visiting a random item URL of the FastAPI app."""
        item_id = random.randint(1, 10)
        self.client.get(f"/items/{item_id}")
