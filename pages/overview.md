# Summary of course content

There are a lot of moving parts in this course, so it may be hard to understand how it all fits together.
This page provides a summary of the frameworks in this course e.g. the stack of tools used. In the figure below we have
provided an overview on how the different tools of the course interacts with each other. The table after the figure
provides a short description of each of the parts.

<figure markdown>
![Overview](../figures/mlops_overview.drawio.png){ width="1000" }
<figcaption>
The MLOps stack in the course. This is just an example of one stack, and depending on your use case you may want to use
a different stack of tools that better fits your needs. Regardless of the stack, the principles of MLOps are the same.
</figcaption>
</figure>

<!-- markdownlint-disable -->
| Framework                                               | Description                                                |
|---------------------------------------------------------|------------------------------------------------------------|
| ![PyTorch](../figures/icons/pytorch.png){ width="50" }  | **PyTorch** is the backbone of our code, it provides the computational engine and the data structures that we need to define our data structures. |
| ![PyTorch Lightning](../figures/icons/lightning.png){ width="50" } | **PyTorch lightning** is a framework that provides a high-level interface to PyTorch. It provides a lot of functionality that we need to train our models, such as logging, checkpointing, early stopping, etc. such that we do not have to implement it ourselves. It also allows us to scale our models to multiple GPUs and multiple nodes. |
| ![Conda](../figures/icons/conda.png){ width="50" } | We control the dependencies and Python interpreter using **Conda** that enables us to construct reproducible virtual environments |
| ![Hydra](../figures/icons/hydra.png){ width="50" } | For configuring our experiments we use **Hydra** that allows us to define a hierarchical configuration structure config files |
| ![Typer](../figures/icons/typer.png){ width="50" } | For creating command line interfaces we can use **Typer** that provides a high-level interface for creating CLIs |
| ![Wandb](../figures/icons/w&b.png){ width="50" } | Using **Weights and Bias** allows us to track and log any values and hyperparameters for our experiments |
| ![Profiler](../figures/icons/profiler.png){ width="50" } | Whenever we run into performance bottlenecks with our code we can use the **Profiler** to find the cause of the bottleneck |
| ![Debugger](../figures/icons/debugger.png){ width="50" } | When we run into bugs in our code we can use the **Debugger** to find the cause of the bug |
| ![Cookiecutter](../figures/icons/cookiecutter.png){ width="50" } | For organizing our code and creating templates we can use **Cookiecutter** |
| ![Docker](../figures/icons/docker.png){ width="50" } | **Docker** is a tool that allows us to create a container that contains all the dependencies and code that we need to run our code |
| ![DVC](../figures/icons/dvc.png){ width="50" } | For controlling the versions of our data and synchronization between local and remote data storage, we can use **DVC** that makes this process easy |
| ![Git](../figures/icons/git.png){ width="50" } | For version control of our code we use **Git** (in complement with GitHub) that allows multiple developers to work together on a shared codebase |
| ![Pytest](../figures/icons/pytest.png){ width="50" } | We can use **Pytest** to write unit tests for our code, to make sure that new changes to the code does break the code base |
| ![Linting](../figures/icons/pep8.png){ width="50" } | For linting our code and keeping a consistent coding style we can use tools such as **Pylint** and **Flake8** that checks our code for common mistakes and style issues |
| ![Actions](../figures/icons/actions.png){ width="50" } | For running our unit tests and other checks on our code in a continuous manner e.g. after we commit and push our code we can use **GitHub actions** that automate this process |
| ![Build](../figures/icons/build.png){ width="50" } | Using **Cloud build** we can automate the process of building our docker images and pushing them to our artifact registry |
| ![Registry](../figures/icons/registry.png){ width="50" } | **Artifact registry** is a service that allows us to store our docker images for later use by other services |
| ![Bucket](../figures/icons/bucket.png){ width="50" } | For storing our data and trained models we can use **Cloud storage** that provides a scalable and secure storage solution |
| ![Engine](../figures/icons/engine.png){ width="50" } | For general compute tasks we can use **Compute engine** that provides a scalable and secure compute solution |
| ![Vertex](../figures/icons/vertex.png){ width="50" } | For training our experiments in a easy and scalable manner we can use **Vertex AI** |
| ![FastAPI](../figures/icons/fastapi.png){ width="50" } | For creating a REST API for our model we can use **FastAPI** that provides a high-level interface for creating APIs |
| ![ONNX](../figures/icons/onnx.png){ width="50" } | For converting our PyTorch model to a format that can be used in production we can use **ONNX** |
| ![Streamlit](../figures/icons/streamlit.png){ width="50" } | For creating a frontend for our model we can use **Streamlit** that provides a high-level interface for creating web applications |
| ![Functions](../figures/icons/functions.png){ width="50" } | For simple deployments of our code we can use **Cloud functions** that allows us to run our code in response to events through simple Python functions |
| ![Run](../figures/icons/run.png){ width="50" } | For more complex deployments of our code we can use **Cloud run** that allows us to run our code in response to events through docker containers |
| ![Locust](../figures/icons/locust.png){ width="50" } | For load testing our deployed model we can use **Locust** |
| ![Monitor](../figures/icons/monitoring.png){ width="50" } | **Cloud monitoring** gives us the tools to keep track of important logs and errors from the other cloud services |
| ![Evidently](../figures/icons/evidentlyai.png){ width="50" } | To monitor whether our deployed model is experiencing any drift, we can use **Evidently AI**, which provides a framework and dashboard for drift monitoring |
| ![Telemetry](../figures/icons/opentelemetry.png){ width="50" } | For monitoring the telemetry of our deployed model we can use **OpenTelemetry** that provides a standard for collecting and exporting telemetry data |
<!-- markdownlint-restore -->
