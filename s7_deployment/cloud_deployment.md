![Logo](../figures/icons/functions.png){ align=right width="130"}
![Logo](../figures/icons/run.png){ align=right width="130"}

# Cloud deployment

---

!!! info "Core Module"

We are now returning to using the cloud. In [this module](../s6_the_cloud/using_the_cloud.md), you should have gone
through the steps of having your code in your GitHub repository to automatically build into a docker container, store
that, store data and pull it all together to make a training run. After the training is completed you should hopefully
have a file stored in the cloud with your trained model weights.

Today's exercises will be about serving those model weights to an end user. We focus on two different ways
of deploying our model: [Google cloud functions](https://cloud.google.com/functions/docs) and
[Google cloud run](https://cloud.google.com/run/docs). Both services are serverless, meaning that you do not have to
manage the server that runs your code.

<figure markdown>
![Image](../figures/gcp_deployment_options.png){ width="1000" }
<figcaption>
GCP in general has 5 core deployment options. We are going to focus on Cloud Functions and Cloud Run, which are two
of the serverless options. In contrast to these two, you have the option to deploy to Kubernetes Engine and Compute
Engine which are more traditional ways of deploying your code. Here you have to manage the underlying infrastructure.
</figcaption>
</figure>

## Cloud Functions

Google Cloud Functions, is the most simple way that we can deploy our code to the cloud. As stated above, it is a
serverless service, meaning that you do not have to worry about the underlying infrastructure. You just write your code
and deploy it. The service is great for small applications that can be encapsulated in **a single script**.

### ❔ Exercises

1. Go to the start page of `Cloud Functions`. Can be found in the sidebar on the homepage or you can just search for it.
    Activate the service in the cloud console or use the following command:

    ```bash
    gcloud services enable cloudfunctions.googleapis.com
    ```

2. Click the `Create Function` button which should take you to a screen like the image below. Make sure it is a 2nd
    Gen function, give it a name, set the server region to somewhere close by and change the authentication policy to
    `Allow unauthenticated invocations` so we can access it directly from a browser. Remember to note down the

    <figure markdown>
    ![Image](../figures/gcp_cloud_functions.png){ width="500" }
    </figure>

3. On the next page, for `Runtime` pick the `Python 3.11` option (or newer). This will make the inline editor show both
    a `main.py` and `requirements.py` file. Look over them and try to understand what they do. Especially, take a
    look at the [functions-framework](https://github.com/GoogleCloudPlatform/functions-framework-python) which is a
    needed requirement of any Cloud function.

    <figure markdown>
    ![Image](../figures/gcp_functions_code.png){ width="500" }
    </figure>

    After you have looked over the files, click the `Deploy` button.

    ??? success "Solution"

        The `functions-framework` is a lightweight, open-source framework for turning Python functions into HTTP
        functions. Any function that you deploy to Cloud Functions must be wrapped in the `@functions_framework.http`
        decorator.

4. Afterwards, the function should begin to deploy. When it is done, you should see ✅. Now let's test it by going to
    the `Testing` tab.

    <figure markdown>
    ![Image](../figures/gcp_test_function.png){ width="800" }
    </figure>

    1.

5. If you know what the application does, it should come as no surprise that it can run without any input. We
    therefore just send an empty request by clicking the `Test The Function` button. Does the function return
    the output you expected? Wait for the logs to show up. What do they show?

    1. What should the `Triggering event` look like in the testing prompt for the program to respond with

        ```txt
        Hallo General Kenobi!
        ```

        Try it out.

        ??? success "Solution"

            The default triggering event is a JSON object with a key `name` and a value. Therefore the triggering event
            should look like this:

            ```json
            {
                "name": "General Kenobi"
            }
            ```

    2. Go to the trigger tab and go to the URL for the application. Execute the API a couple of times. How can you
        change the URL to make the application respond with the same output as above?

        ??? success "Solution"

            You can change the URL to include a query parameter `name` with the value `General Kenobi`. For example

            ```txt
            https://us-central1-my-personal-mlops-project.cloudfunctions.net/function-3?name=General%20Kanobi
            ```

            where you would need to replace everything before the `?` with your URL.

    3. Click on the metrics tab. You should hopefully see it being populated with a few datapoints. Identify what each
        panel is showing.

        ??? success "Solution"

            *  Invocations/Second: The number of times the function is invoked per second
            *  Execution time (ms): The time it takes for the function to execute in milliseconds
            *  Memory usage (MB): The memory usage of the function in MB
            *  Instance count (instances): The number of instances that are running the function

    4. Check out the logs tab. You should see that your application has already been invoked multiple times. Also, try
        to execute this command in a terminal:

        ```bash
        gcloud functions logs read
        ```

6. Next, we are going to create our own application that takes some input so we can try to send it requests. We provide
    a very simple script to get started.

    !!! example "Simple script"

        ```python linenums="1" title="sklearn_cloud_functions.py"
        --8<-- "s7_deployment/exercise_files/sklearn_cloud_functions.py"
        ```

    1. Figure out what the script does and run the script. This should create a file with a trained model.

        ??? success "Solution"

            The file trains a simple KNN model on the iris dataset and saves it to a file called `model.pkl`.

    2. Next, create a storage bucket and upload the model file to the bucket. Try to do this using the `gsutil` command
        and check afterward that the file is in the bucket.

        ??? success "Solution"

            ```bash
            gsutil mb gs://<bucket-name>  # mb stands for make bucket
            gsutil cp <file-name> gs://<bucket-name>  # cp stands for copy
            ```

    3. Create a new cloud function with the same initial settings as the first one, e.g. `Python 3.11` and `HTTP`. Then
        implement in the `main.py` file code that:

        * Loads the model from the bucket
        * Takes a request with a list of integers as input
        * Returns the prediction of the model

        In addition to writing the `main.py` file, you also need to fill out the `requirements.txt` file. You need at
        least three packages to run the application. Remember to also change the `Entry point` to the name of your
        function. If your deployment fails, try to go to the `Logs Explorer` page in `gcp` which can help you identify
        why.

        ??? success "Solution"

            The main script should look something like this:

            ```python linenums="1" title="main.py"
            --8<-- "s7_deployment/exercise_files/sklearn_main_function.py"
            ```

            And, the requirement file should look like this:

            ```txt
            --8<-- "s7_deployment/exercise_files/sklearn_function_main_requirements.txt"
            ```

            importantly make sure that you are using the same version of `scikit-learn` as you used when you trained the
            model. Else when trying to load the model you will most likely get an error.

    4. When you have successfully deployed the model, try to make predictions with it. What should the request
        look like?

        ??? success "Solution"

            It depends on how exactly you have chosen to implement the `main.py`. But for the provided solution, the
            payload should look like this:

            ```json
            {
                "data": [1, 2, 3, 4]
            }
            ```

            with the corresponding `curl` command:

            ```bash
            curl -X POST \
                https://your-cloud-function-url/knn_classifier \
                -H "Content-Type: application/json" \
                -d '{"input_data": [5.1, 3.5, 1.4, 0.2]}'
            ```

7. Let's try to figure out how to do the above deployment using `gcloud` instead of the console UI. The relevant command
    is [gcloud functions deploy](https://cloud.google.com/functions/docs/create-deploy-http-python). For this function
    to work you will need to put the `main.py` and `requirements.txt` in a separate folder. Try to execute the command
    to successfully deploy the function.

    ??? success "Solution"

        ```bash
        gcloud functions deploy <func-name> \
            --gen2 --runtime python311 --trigger-http --source <folder> --entry-point knn_classifier
        ```

        where you need to replace `<func-name>` with the name of your function and `<folder>` with the path to the
        folder containing the `main.py` and `requirements.txt` files.

8. (Optional) You can finally try to redo the exercises by deploying a Pytorch application. You will essentially
    need to go through the same steps as the sklearn example, including uploading a trained model to storage and
    writing a cloud function that loads it and returns some output. You are free to choose whatever Pytorch model you
    want.

## Cloud Run

Cloud functions are great for simple deployments, that can be encapsulated in a single script with only simple
requirements. However, they do not scale with more advanced applications that may depend on multiple programming
languages. We are already familiar with how we can deal with this through containers and
[Cloud Run](https://cloud.google.com/run/docs/overview/what-is-cloud-run) is the corresponding service in GCP for
deploying containers.

### ❔ Exercises

1. We are going to start locally by developing a small app that we can deploy. We provide two small examples to choose
    from: first is a small FastAPI app consisting of this
    [.py file](https://github.com/SkafteNicki/dtu_mlops/tree/main/s7_deployment/exercise_files/simple_fastapi_app.py)
    and this
    [dockerfile](https://github.com/SkafteNicki/dtu_mlops/tree/main/s7_deployment/exercise_files/simple_fastapi_app.dockerfile)
    . Secondly a small [streamlit](https://streamlit.io/) application consisting of just this
    [dockerfile](https://github.com/SkafteNicki/dtu_mlops/tree/main/s7_deployment/exercise_files/streamlit_app.dockerfile)
    . You are free to choose which application to work with.

    1. Start by going over the files belonging to your choice app and understand what it does.

    2. Next build the docker image belonging to the app

        ```bash
        docker build -f <dockerfile> . -t gcp_test_app:latest
        ```

    3. Next tag and push the image to your container registry

        ```bash
        docker tag gcp_test_app gcr.io/<project-id>/gcp_test_app
        docker push gcr.io/<project-id>/gcp_test_app
        ```

        afterwards check you container registry to check that you have successfully pushed the image.

2. Next go to `Cloud Run` in the cloud console an enable the service

3. Click the `Create Service` button which should bring you to a page similar to the one below

    <figure markdown>
    ![Image](../figures/gcp_run.PNG){ width="1000" }
    </figure>

    Do the following:
    * Click the select button, which will bring up all build containers and pick the one you want to deploy. In the
        future you probably want to choose the *Continuously deploy new revision from a source repository* such that a new
        version is always deployed when a new container is build.
    * Hereafter, give the service a name and select the region. We recommend do choose a region close to you, however
        it does not really matter that much for our use case
    * Set the authentication method to *Allow unauthenticated invocations* such that we can call it without
        providing credentials. In the future you may only set that authenticated invocations are allowed.
    * Expand the *Container, Connections, Security* tab and edit the port such that it matches the port exposed in your
        chosen application.

    Finally, click the create button and wait for the service to be deployed (may take some time).

    !!! warning "Common problems"

        If you get an error saying
        *The user-provided container failed to start and listen on the port defined by the PORT environment variable.*
        there are two common reasons for this:

        1. You need to add an `EXPOSE` statement in your docker container:

            ```dockerfile
            EXPOSE 8080
            CMD exec uvicorn my_application:app --port 8080 --workers 1 main:app
            ```

            and make sure that your application is also listening on that port. If you hard code the port in your
            application (as in above code) it is best to set it 8080 which is the default port for cloud run.
            Alternatively, a better approach is to set it to the `$PORT` environment variable which is set by cloud run
            and can be accessed in your application:

            ```dockerfile
            EXPOSE $PORT
            CMD exec uvicorn my_application:app --port $PORT --workers 1 main:app
            ```

            If you do this, and then want to run locally you can run it as:

            ```bash
            docker run -p 8080:8080 -e PORT=8080 <image-name>:<image-tag>
            ```

        2. If you are serving a large machine learning model, it may also be that your deployed container is running
            out of memory. You can try to increase the memory of the container by going to the *Edit container* and
            the *Resources* tab and increase the memory.

4. If you manage to deploy the service you should see a image like this:

    <figure markdown>
    ![Image](../figures/gcp_run2.PNG){ width="1000" }
    </figure>

    You can now access you application by clicking url. This will access the root of your application, so you may need
    to add `/` or `/<path>` to the url depending on how the app works.

5. Everything we just did to deploy an container can be reproduced using the following command:

    ```bash
    gcloud run deploy $APP --image $TAG --platform managed --region $REGION --allow-unauthenticated
    ```

    and checked using these two commands

    ```bash
    gcloud run services list
    gcloud run services describe $APP --region $REGION
    ```

    feel free to experiment doing the deployment from the command line.

6. Instead of deploying our docker container using the UI or command line, which is a manual operation, we can do it
    in a continuous manner by using `cloudbuild.yaml` file we learned about in the previous section. We just need to add
    a new step to the file. We provide an example

    ```yaml
    steps:
    # Build the container image
    - name: 'gcr.io/cloud-builders/docker'
      args: ['build', '-t', 'gcr.io/$PROJECT_ID/<container-name>:latest', '.'] #(1)!
    # Push the container image to Container Registry
    - name: 'gcr.io/cloud-builders/docker'
      args: ['push', 'gcr.io/$PROJECT_ID/<container-name>:latest']
    # Deploy container image to Cloud Run
    - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
      entrypoint: gcloud
      args:
      - 'run'
      - 'deploy'
      - '<service-name>'
      - '--image'
      - 'gcr.io/$PROJECT_ID/<container-name>:latest'
      - '--region'
      - '<region>'
    ```

    1. This line assume you are standing in the root of your repository and is trying to build the docker image
        specified in a file called `Dockerfile` and tag it with the name `gcr.io/$PROJECT_ID/my_deployment:latest`.
        Therefore if you want to point to another dockerfile you need to add `-f` option to the command. For example
        if you want to point to a `my_app/my_serving_app.dockerfile` you need to change the line to

        ```yaml
        args: ['build', '-f', 'my_app/my_serving_app.dockerfile', '-t', 'gcr.io/$PROJECT_ID/my_deployment:lates', '.']
        ```

    where you need to replace `<container-name>` with the name of your container, `<service-name>` with the name of the
    service you want to deploy and `<region>` with the region you want to deploy to. Afterwards you need to setup a
    trigger (or reuse the one you already have) to build the container and deploy it to cloud run. Confirm that this
    works by making a change to your application and pushing it to GitHub and see if the application is updated
    continuously. For help you can look [here](https://cloud.google.com/build/docs/deploying-builds/deploy-cloud-run)
    for help. If you succeeded, congratulations you have now setup a continuous deployment pipeline.

That ends the exercises on deployment. The exercises above is just a small taste of what deployment has to offer. In
both sections we have explicitly chosen to work with *serverless* deployments. But what if you wanted to do the
opposite e.g. being the one in charge of the management of the cluster that handles the deployed services? If you are
really interested in taking deployment to the next level should get started on *kubernetes* which is the de-facto
open-source container orchestration platform that is being used in production environments. If you want to deep dive we
recommend starting [here](https://cloud.google.com/ai-platform/pipelines/docs) which describes how to make pipelines
that are a necessary component before you start to
[create](https://cloud.google.com/ai-platform/pipelines/docs/configure-gke-cluster) your own kubernetes cluster.
