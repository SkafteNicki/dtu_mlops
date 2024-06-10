![Logo](../figures/icons/streamlit.png){ align=right width="130"}

# Frontend

If you have gone over the [deployment module](../s7_deployment/README.md) you should be at the point where you have a
machine learning model running in the cloud. The model can be interacted with by sending HTTP requests to the API
endpoint. In general we refer to this as the *backend* of the application. It is the part of our application that are
behind-the-scences that the user does not see and it is not really that user-friendly. Instead we want to create a
*frontend* that the user can interact with in a more user-friendly way. This is what we will be doing in this module.

Another point of splitting our application into a frontend and a backend has to do with scalability. If we have a lot
of users interacting with our application, we might want to scale only the backend and not the frontend, because that
is the part that will be running our heavy machine learning model. In general dividing a application into smaller pieces
are the pattern that is used in [microservice architectures](https://martinfowler.com/articles/microservices.html).

<figure markdown>
![Image](../figures/different_architechtures.png){ width="800" }
<figcaption>
In monollithic applications everything the user may be requesting of our application is handled by a single process/
container. In microservice architectures the application is split into smaller pieces that can be scaled independently.
This also leads to easier maintainability and faster development.
<a
href="https://www.researchgate.net/figure/Comparison-between-monolithic-and-microservices-architectures-for-an-application-that_fig5_355361563">
Image Credit.
</a>
</figcaption>
</figure>

Frontends have for the longest time been created using HTML, CSS and JavaScript. This is still the case, but there are
now a lot of frameworks that can help us create a frontend in Python:

* [Django](https://www.djangoproject.com/)
* [Reflex](https://reflex.dev/)
* [Streamlit](https://streamlit.io/)
* [Bokeh](http://bokeh.org/)
* [Gradio](https://www.gradio.app/)

In this module we will be looking at `streamlit`. `streamlit` is a easy to use framework that allows us to create
interactive web applications in Python. It is not at all as powerful as a framework like `Django`, but it is very easy
to get started with and it is very easy to integrate with our machine learning models.

### ❔ Exercises

In these exercises we go through the process of setting up a backend using `fastapi` and a frontend using `streamlit`,
containerizing both applications and then deploying them to the cloud. We have already created an example of this
which can be found in the `samples/frontend_backend` folder.

1. Start by installing `streamlit`

    ```bash
    pip install streamlit
    ```

    and run `streamlit hello` afterwards to check that everything works as expected.

2. Lets start by creating the backend application in a `backend.py` file. You can use essentially any backend you want,
    but we will be using a simple imagenet classifier that we have created in the `samples/frontend_backend/backend`
    folder.

    1. Create a new file called `backend.py` and copy the code from the imagenet classifier into it. 

    2. Run the backend using `uvicorn`

        ```bash
        uvicorn backend:app --reload
        ```

    3. Test the backend by sending a request to the `/predict` endpoint

        ```bash
        curl -X 'POST' \
          'http://
            localhost:8000/predict' \
            -H 'accept: application/json' \
            -H 'Content-Type: application/json' \
            -d '{
            "url": "https://upload.wikimedia.org/wikipedia/commons/4/4e/Pleiades_large.jpg"
            }'
        ```

    4. Containerize the backend into a file called `backend.dockerfile`.

        ```dockerfile
        FROM python:3.8-slim

        WORKDIR /app

        COPY requirements.txt .

        RUN pip install --no-cache-dir -r requirements.txt

        COPY backend.py .

        CMD ["uvicorn", "backend:app", "--host"]
        ```

    5. Build the backend image

        ```bash
        docker build -t backend -f backend.dockerfile .
        ```

    6. Run the backend image

        ```bash
        docker run -p 8000:8000 backend
        ```

    7. Test the backend by sending a request to the `/predict` endpoint

        ```bash
        curl -X 'POST' \
          'http://
            localhost:8000/predict' \
            -H 'accept: application/json' \
            -H 'Content-Type: application/json' \
            -d '{
            "url": "https://upload.wikimedia.org/wikipedia/commons/4/4e/Pleiades_large.jpg"
            }'
        ```

    8. Deploy the backend to Cloud run using the `gcloud` command

        ```bash
        gcloud builds submit --tag gcr.io/PROJECT_ID/backend
        gcloud run deploy --image gcr.io/PROJECT_ID/backend --platform managed
        ```

    9. Test the backend by sending a request to the `/predict` endpoint

        ```bash
        curl -X 'POST' \
          'https://
            PROJECT_ID.REGION.r.appspot.com/predict' \
            -H 'accept: application/json' \
            -H 'Content-Type: application/json' \
            -d '{
            "url": "https://upload.wikimedia.org/wikipedia/commons/4/4e/Pleiades_large.jpg"
            }'
        ```

3. With the backend taken care of lets now write our frontend. Our frontend just needs to be a "nice" interface to our
    backend. Its main functionality will be to send a request to the backend and display the result.
    [streamlit documentation](https://docs.streamlit.io/library/api-reference)

    1. Create a new file called `frontend.py` and copy the code from the frontend in the `samples/frontend_backend/frontend`
        folder. You can choose to use any frontend you want, but we will be using `streamlit`.

    2. We need to make sure that the frontend knows where the backend is located, and we want that to happen 
        automatically so we do not have to hardcode the URL into our frontend. We can do this by using the 
        Python SDK for Google Cloud Run. The following code snippet shows how to get the URL of the backend service:

        ```python
        from google.cloud import run_v2
        import streamlit as st

        @st.cache_resource  # (1)!
        def get_backend_url():
            """Get the URL of the backend service."""
            parent = "projects/my-personal-mlops-project/locations/europe-west1"
            client = run_v2.ServicesClient()
            services = client.list_services(parent=parent)
            for service in services:
                if service.name.split("/")[-1] == "production-model":
                    return service.uri

        BACKEND = get_backend_url()
        if BACKEND is None:
            raise ValueError("Backend service not found")
        ```

        1. :man_raising_hand: The `st.cache_resource` is a decorator that tells `streamlit` to cache the result of the
            function. This is useful if the function is expensive to run and we want to avoid running it multiple times.

    2. Run the frontend using `streamlit`

        ```bash
        streamlit run frontend.py
        ```

    3. Containerize the frontend into a file called `frontend.dockerfile`.

        ```dockerfile
        FROM python:3.8-slim

        WORKDIR /app

        COPY requirements.txt .

        RUN pip install --no-cache-dir -r requirements.txt

        COPY frontend.py .

        CMD ["streamlit", "run", "frontend.py"]
        ```

    4. Build the frontend image

        ```bash
        docker build -t frontend -f frontend.dockerfile .
        ```

    5. Run the frontend image

        ```bash
        docker run -p 8501:8501 frontend
        ```

    6. Test the frontend by opening a browser and going to `http://localhost:8501`


    7. Deploy the frontend to Cloud run using the `gcloud` command

        ```bash
        gcloud builds submit --tag gcr.io/PROJECT_ID/frontend
        gcloud run deploy --image gcr.io/PROJECT_ID/frontend --platform managed
        ```

    8. Test the frontend by opening a browser and going to `https://PROJECT_ID.REGION.r.appspot.com`

4. If you have gotten this far you have successfully created a frontend and a backend and deployed them to the cloud.
    Finally, it may be worth it to load test your application to see how it performs under load. Write a locust file
    which is covered in [this module](../s7_deployment/testing_apis.md) and run it against your frontend. Make sure
    that it can handle the load you expect it to handle.

This ends the exercises for this module