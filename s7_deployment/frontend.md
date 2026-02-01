![Logo](../figures/icons/streamlit.png){ align=right width="130"}

# Frontend

If you have gone over the [deployment module](../s7_deployment/README.md) you should be at the point where you have a
machine learning model running in the cloud. The model can be interacted with by sending HTTP requests to the API
endpoint. In general we refer to this as the *backend* of the application. It is the part of our application that is
behind-the-scenes that the user does not see and is not really that user-friendly. Instead we want to create a
*frontend* that the user can interact with in a more user-friendly way. This is what we will be doing in this module.

Another point of splitting our application into a frontend and a backend has to do with scalability. If we have a lot
of users interacting with our application, we might want to scale only the backend and not the frontend, because that
is the part that will be running our heavy machine learning model. In general dividing an application into smaller pieces
is the pattern that is used in [microservice architectures](https://martinfowler.com/articles/microservices.html).

<figure markdown>
![Image](../figures/different_architechtures.png){ width="800" }
<figcaption>
In monollithic applications everything the user may be requesting of our application is handled by a single process/
container. In microservice architectures the application is split into smaller pieces that can be scaled independently.
This also leads to easier maintainability and faster development.
</figcaption>
</figure>

Frontends have for the longest time been created using HTML, CSS and JavaScript. This is still the case, but there are
now a lot of frameworks that can help us create a frontend in Python:

* [Django](https://www.djangoproject.com/)
* [Reflex](https://reflex.dev/)
* [Streamlit](https://streamlit.io/)
* [Bokeh](http://bokeh.org/)
* [Gradio](https://www.gradio.app/)

In this module we will be looking at `streamlit`. `streamlit` is an easy-to-use framework that allows us to create
interactive web applications in Python. It is not at all as powerful as a framework like `Django`, but it is very easy
to get started with and it is very easy to integrate with our machine learning models.

## ❔ Exercises

In these exercises we go through the process of setting up a backend using `fastapi` and a frontend using `streamlit`,
containerizing both applications and then deploying them to the cloud. We have already created an example of this
which can be found in the `samples/frontend_backend` folder.

1. Let's start by creating the backend application in a `backend.py` file. You can use essentially any backend you want,
    but we will be using a simple imagenet classifier that we have created in the `samples/frontend_backend/backend`
    folder.

    1. Create a new file called `backend.py` and implement a FastAPI interface with a single `/predict` endpoint that
        takes an image as input and returns the predicted class (and probabilities) of the image.

        ??? success "Solution"

            ```python linenums="1" title="backend.py"
            --8<-- "samples/frontend_backend/backend.py"
            ```

    2. Run the backend using `uvicorn`.

        ```bash
        uvicorn backend:app --reload
        ```

    3. Test the backend by sending a request to the `/predict` endpoint, preferably using the `curl` command.

        ??? success "Solution"

            In this example we are sending a request to the `/predict` endpoint with a file called `my_cat.jpg`. The
            response should be "tabby cat" for the solution we have provided.

            ```bash
            curl -X 'POST' \
                'http://127.0.0.1:8000/classify/' \
                -H 'accept: application/json' \
                -H 'Content-Type: multipart/form-data' \
                -F 'file=@my_cat.jpg;type=image/jpeg'
            ```

    4. We are going to use a single `pyproject.toml` file with dependency groups to separate the backend and frontend
        dependencies. This allows us to keep all dependencies in one file while still being able to install only what
        we need for each service. Try making used of the `[dependency-groups]` field for this. Create the file now.

        ??? success "Solution"

            ```toml linenums="1" title="pyproject.toml"
            --8<-- "samples/frontend_backend/pyproject.toml"
            ```

            The `[dependency-groups]` section allows us to define separate groups of dependencies. We can then install
            only the dependencies we need using `uv sync --group <group-name>`. This is particularly useful when
            containerizing applications, as we can keep our Docker images smaller by only installing the dependencies
            needed for that specific service.

    5. Containerize the backend into a file called `backend.dockerfile`.

        ??? success "Solution"

            ```dockerfile linenums="1" title="backend.dockerfile"
            --8<-- "samples/frontend_backend/backend.dockerfile"
            ```

            Notice how we use `uv sync --group backend` to install only the backend dependencies from our unified
            `pyproject.toml` file. The `--no-install-project` flag tells uv not to install the project itself (since
            we don't have a package to install), and `--no-dev` skips development dependencies.

    6. Build the backend image

        ```bash
        docker build -t backend:latest -f backend.dockerfile .
        ```

    7. Recheck that the backend works by running the image in a container

        ```bash
        docker run --rm -p 8000:8000 -e "PORT=8000" backend
        ```

        and test that it works by sending a request to the `/predict` endpoint.

    8. Deploy the backend to Cloud run using the `gcloud` command.

        ??? success "Solution"

            Assuming that we have created an artifact registry called `frontend_backend` we can deploy the backend to
            Cloud Run using the following commands:

            ```bash
            docker tag \
                backend:latest \
                <region>-docker.pkg.dev/<project>/frontend-backend/backend:latest
            docker push \
                <region>.pkg.dev/<project>/frontend-backend/backend:latest
            gcloud run deploy backend \
                --image=europe-west1-docker.pkg.dev/<project>/frontend-backend/backend:latest \
                --region=europe-west1 \
                --platform=managed \
            ```

            where `<region>` and `<project>` should be replaced with the appropriate values.

    9. Finally, test that the deployed backend works as expected by sending a request to the `/predict` endpoint.

        ??? success "Solution"

            In this solution we are first extracting the url of the deployed backend and then sending a request to the
            `/predict` endpoint.

            ```bash
            export MYENDPOINT=$(gcloud run services describe backend --region=<region> --format="value(status.url)")
            curl -X 'POST' \
                $MYENDPOINT/predict \
                -H 'accept: application/json' \
                -H 'Content-Type: multipart/form-data' \
                -F 'file=@my_cat.jpg;type=image/jpeg'
            ```

2. With the backend taken care of let's now write our frontend. Our frontend just needs to be a "nice" interface to our
    backend. Its main functionality will be to send a request to the backend and display the result
    ([streamlit documentation](https://docs.streamlit.io/library/api-reference)).

    1. Start by installing `streamlit`.

        ```bash
        uv add --group frontend streamlit
        ```

    2. Now create a file called `frontend.py` and implement a streamlit application. You can design it however you want,
        but we recommend that the following can be done in the frontend:

        1. Have a file uploader that allows the user to upload an image

        2. Display the image that the user uploaded

        3. Have a button that sends the image to the backend and displays the result

        For now just assume that an environment variable called `BACKEND` is available that contains the URL of the
        backend. We will in the next step show how to get this URL automatically.

        ??? success "Solution"

            ```python linenums="1" title="frontend.py"
            --8<-- "samples/frontend_backend/frontend.py"
            ```

    3. We need to make sure that the frontend knows where the backend is located, and we want that to happen
        automatically so we do not have to hardcode the URL into our frontend. We can do this by using the
        Python SDK for Google Cloud Run. The following code snippet shows how to get the URL of the backend service
        or fall back on an environment variable if the service is not found.

        ```python
        from google.cloud import run_v2
        import streamlit as st

        @st.cache_resource  # (1)!
        def get_backend_url():
            """Get the URL of the backend service."""
            parent = "projects/<project>/locations/<region>"
            client = run_v2.ServicesClient()
            services = client.list_services(parent=parent)
            for service in services:
                if service.name.split("/")[-1] == "production-model":
                    return service.uri
            name = os.environ.get("BACKEND", None)
            return name
        ```

        1. :man_raising_hand: The `st.cache_resource` is a decorator that tells `streamlit` to cache the result of the
            function. This is useful if the function is expensive to run and we want to avoid running it multiple times.

        Add the above code snippet to the top of your `frontend.py` file and replace `<project>` and `<region>` with the
        appropriate values. You will need to install google-cloud-run using `uv add --group frontend google-cloud-run`
        to be able to use the code snippet.

    4. Run the frontend using `streamlit`.

        ```bash
        uv run streamlit run frontend.py
        ```

    5. Update the `pyproject.toml` file from earlier with a `frontend` group under the `[dependency-groups]` section
        that contains the `streamlit` dependency and any other dependencies you may need for the frontend. Make sure
        that you can install the frontend dependencies separately and all dependencies together.

        ```bash
        uv sync --group frontend  # only frontend dependencies
        uv sync --all-groups      # all dependencies
        ```

    6. Containerize the frontend into a file called `frontend.dockerfile`.

        ??? success "Solution"

            ```dockerfile linenums="1" title="frontend.dockerfile"
            --8<-- "samples/frontend_backend/frontend.dockerfile"
            ```

            Similar to the backend, we use `uv sync --group frontend` to install only the frontend dependencies,
            keeping the Docker image as small as possible.

    7. Build the frontend image.

        ```bash
        docker build -t frontend:latest -f frontend.dockerfile .
        ```

    8. Run the frontend image

        ```bash
        docker run --rm -p 8001:8001 -e "PORT=8001" frontend
        ```

        and check in your web browser that the frontend works as expected.

    9. Deploy the frontend to Cloud run using the `gcloud` command.

        ??? success "Solution"

            Assuming that we have created an artifact registry called `frontend_backend` we can deploy the frontend to
            Cloud Run using the following commands:

            ```bash
            docker tag frontend:latest \
                <region>-docker.pkg.dev/<project>/frontend-backend/frontend:latest
            docker push <region>.pkg.dev/<project>/frontend-backend/frontend:latest
            gcloud run deploy frontend \
                --image=europe-west1-docker.pkg.dev/<project>/frontend-backend/frontend:latest \
                --region=europe-west1 \
                --platform=managed \
            ```

    10. Test that the frontend works as expected by opening the URL of the deployed frontend in your web browser.

3. (Optional) If you have gotten this far you have successfully created a frontend and a backend and deployed them to
    the cloud. Finally, it may be worth load testing your application to see how it performs under load. Write a
    locust file which is covered in [this module](../s7_deployment/testing_apis.md) and run it on your frontend.
    Make sure that it can handle the load you expect it to handle.

4. (Optional) Feel free to experiment further with streamlit and see what you can create. For example, you can try to
    create an option for the user to upload a video and then display the video with the predicted class overlaid on
    top of the video.

## 🧠 Knowledge check

1. We have used dependency groups to separate the frontend and backend dependencies in a single `pyproject.toml` file.
    Why is this approach beneficial compared to having completely separate files?

    ??? success "Solution"

        Using dependency groups provides several benefits:

        1. **Single source of truth**: All project dependencies are in one place, making it easier to manage and version
           control.
        2. **Selective installation**: We can still install only what we need for each service using
           `uv sync --group <group-name>`, keeping Docker images small.
        3. **Easier maintenance**: When dependencies need updating, we only need to edit one file instead of multiple.
        4. **Flexibility**: We can easily install all dependencies for local development using `uv sync --all-groups`,
           or install specific groups for production deployments.
        5. **Better organization**: Related dependencies are grouped logically, making it clear which dependencies belong
           to which part of the application.

        The separation of concerns is still maintained (backend vs frontend dependencies), but without the overhead of
        managing multiple files. This is particularly valuable as projects grow and dependency management becomes more
        complex.

This ends the exercises for this module.
