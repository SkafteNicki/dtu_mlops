![Logo](../figures/icons/functions.png){ align=right width="130"}
![Logo](../figures/icons/run.png){ align=right width="130"}

# Testing APIs

---

!!! info "Core Module"

The is in general two things that we want to test when we are working with APIs:

* Does the API work as intended? e.g. for a given input, does it return the expected output?
* Can the API handle the expected load? e.g. if we send 1000 requests per second, does it crash?

In this module we go over how to do each of them:

## Testing for functionality

Similar to when we wrote unit tests for our code back in [this module](../s5_continuous_integration/unittesting.md) we
can also write tests for our API. The difference is that instead of testing individual functions, we are testing the
entire API. As always we recommend implementing the tests in a separate folder called `tests`:

```plaintext
my_fastapi_app/
|-- app/
|   |-- main.py
|   |-- ...
|-- tests/
|   |-- test_main.py
|   |-- ...
```

Before you get started with the exercises we recommend that you start by defining an environment variable that contains
the endpoint of your API. This will make it easier to write the tests. You can do this by running the following command:

=== "Windows"

    ```bash
    for /f "delims=" %i in ^
    ('gcloud run services describe <name> --region=<region> --format="value(status.url)"') do set MYENDPOINT=%i

=== "Mac/Linux"

    ```bash
    export MYENDPOINT=$(gcloud run services describe <name> --region=<region> --format="value(status.url)")
    ```

where you replace `<name>` and `<region>` with the name of your service and the region it is deployed in.

### ❔ Exercises

```python
from bentoml.testing.server import host_bento
```

## Testing for load

Something about load testing

### ❔ Exercises

1. Install `locust`

    ```bash
    pip install locust
    ```

2. Read this [guide](https://docs.locust.io/en/stable/writing-a-locustfile.html) that explains how to write a
    `locustfile.py`. Afterwards, try writing a `locustfile.py` for your application:

    ```python
    import time
    from locust import HttpUser, task, between

    class QuickstartUser(HttpUser):
        wait_time = between(1, 5)

        @task
        def hello_world(self):
            self.client.get("/hello")
            self.client.get("/world")

        @task(3)
        def view_items(self):
            for item_id in range(10):
                self.client.get(f"/item?id={item_id}", name="/item")
                time.sleep(1)

        def on_start(self):
            self.client.post("/login", json={"username":"foo", "password":"bar"})
    ```

3. Then try to run the `locust` command:

    ```bash
    locust -f locustfile.py
    ```
