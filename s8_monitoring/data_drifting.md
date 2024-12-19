
![Logo](../figures/icons/evidentlyai.png){align=right width="130"}

# Data drifting

---

Data drifting is one of the core reasons for model accuracy degrades over time in production. For machine learning
models, data drift is the change in model input data that leads to model performance degradation. In practical terms,
this means that the model is receiving input that is outside the scope that it was trained on, as seen in the figure
below. This shows that the underlying distribution of a particular feature has slowly been increasing in value over
two years

<figure markdown>
![Image](../figures/data_drift.png){ width="700" }
<figcaption>
<a href="https://www.picsellia.com/post/what-is-data-drift-and-how-to-detect-it-with-mlops"> Image credit </a>
</figcaption>
</figure>

In some cases, it may be that if you normalize some feature in a better way that you are able to generalize your model
better, but this is not always the case. The reason for such a drift is commonly some external factor that you
essentially have no control over. That really only leaves you with one option: retrain your model on the newly received
input features and deploy that model to production. This process is probably going to repeat over the lifetime of your
application if you want to keep it up-to-date with the real world.

<figure markdown>
![Image](../figures/retrain_model.png){ width="700" }
<figcaption>
<a href="https://www.evidentlyai.com/blog/machine-learning-monitoring-data-and-concept-drift"> Image credit </a>
</figcaption>
</figure>

We have now come up with a solution to the data drift problem, but there is one important detail that we have not taken
care of: When we should actually trigger the retraining? We do not want to wait around for our model performance to
degrade, thus we need tools that can detect when we are seeing a drift in our data.

## ❔ Exercises

For these exercises we are going to use the framework [Evidently](https://github.com/evidentlyai/evidently) developed by
[EvidentlyAI](https://www.evidentlyai.com). Evidently currently supports both detection for both regression and
classification models. The exercises are in large taken from
[here](https://docs.evidentlyai.com/get-started/hello-world) and in general we recommend if you are in doubt about an
exercise to look at the [docs](https://docs.evidentlyai.com/) for API and examples (their documentation can be a bit
lacking sometimes, so you may also have to dive into the source code).

Additionally, we want to stress that data drift detection, concept drift detection etc. is still an active field of
research and therefore exist multiple frameworks for doing this kind of detection. In addition to Evidently,
we can also mention [NannyML](https://github.com/NannyML/nannyml), [WhyLogs](https://github.com/whylabs/whylogs) and
[deepcheck](https://github.com/deepchecks/deepchecks).

1. Start by installing Evidently

    ```python
    pip install evidently
    ```

    You will also need `scikit-learn` and `pandas` installed if you do not already have it.

2. Hopefully you have already gone through session [S7 on deployment](../s7_deployment/README.md). As part of the
    deployment sections you should have developed an application that can classify the
    [iris dataset](https://archive.ics.uci.edu/ml/datasets/iris), based on a model trained by this
    [script](https://github.com/SkafteNicki/dtu_mlops/tree/main/s7_deployment/exercise_files/sklearn_cloud_functions.py)
    . We are going to convert this into a FastAPI application for the purpose here:

    1. Convert your GCP function into a FastAPI application. The appropriate `curl` command should look something like
        this:

        ```bash
        curl -X 'POST' \
            'http://127.0.0.1:8000/iris_v1/?sepal_length=1.0&sepal_width=1.0&petal_length=1.0&petal_width=1.0' \
            -H 'accept: application/json' \
            -d ''
        ```

        and the response body should look like this:

        ```json
        {
            "prediction": "Iris-Setosa",
            "prediction_int": 0
        }
        ```

        We have implemented a solution in this
        [file](https://github.com/SkafteNicki/dtu_mlops/tree/main/s8_monitoring/exercise_files/iris_fastapi.py)
        (called v1) if you need help.

    2. Next we are going to add some functionality to our application. We need to add that the input for the user is
        saved to a database whenever our application is called. However, to not slow down the response to our user we
        want to implement this as an *background task*. A background task is a function that should be executed after
        the user have got their response. Implement a background task that save the user input to a database implemented
        as a simple `.csv` file. You can read more about background tasks
        [here](https://fastapi.tiangolo.com/tutorial/background-tasks/). The header of the database should look
        something like this:

        ```csv
        time, sepal_length, sepal_width, petal_length, petal_width, prediction
        2022-12-28 17:24:34.045649, 1.0, 1.0, 1.0, 1.0, 1
        2022-12-28 17:24:44.026432, 2.0, 2.0, 2.0, 2.0, 1
        ...
        ```

        thus both input, timestamp and predicted value should be saved. We have implemented a solution in this
        [file](exercise_files/iris_fastapi.py) (called v2) if you need help.

    3. Call you API a number of times to generate some dummy data in the database.

3. Create a new `data_drift.py` file where we are going to implement the data drifting detection and reporting. Start
    by adding both the real iris data and your generated dummy data as pandas dataframes.

    ```python
    import pandas as pd
    from sklearn import datasets
    reference_data = datasets.load_iris(as_frame=True).frame
    current_data = pd.read_csv('prediction_database.csv')
    ```

    if done correctly you will most likely end up with two dataframes that look like

    ```txt
    # reference_data
    sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  target
    0                  5.1               3.5                1.4               0.2       0
    1                  4.9               3.0                1.4               0.2       0
    ...
    148                6.2               3.4                5.4               2.3       2
    149                5.9               3.0                5.1               1.8       2
    [150 rows x 5 columns]

    # current_data
    time                         sepal_length   sepal_width   petal_length   petal_width   prediction
    2022-12-28 17:24:34.045649   1.0            1.0            1.0           1.0           1
    ...
    2022-12-28 17:24:34.045649   1.0            1.0            1.0           1.0           1
    [10 rows x 5 columns]
    ```

    Standardize the dataframes such that they have the same column names and drop the time column from the
    `current_data` dataframe.

4. We are now ready to generate some reports about data drifting:

    1. Try executing the following code:

        ```python
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference, current_data=current)
        report.save_html('report.html')
        ```

        and open the generated `.html` page. What does it say about your data? Have it drifted? Make sure to poke
        around to understand what the different plots are actually showing.

    2. Data drifting is not the only kind of reporting evidently can make. We can also get reports on the data quality.
        Try first adding a few `Nan` values to your reference data. Secondly, try changing the report to

        ```python
        from evidently.metric_preset import DataDriftPreset, DataQualityPreset
        report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
        ```

        and re-run the report. Checkout the newly generated report. Again go over the generated plots and make sure that
        it picked up on the missing values you just added.

    3. The final report present we will look at is the `TargetDriftPreset`. Target drift means that our model is
        over/under predicting certain classes e.g. or general terms the distribution of predicted values differs from
        the ground true distribution of targets. Try adding the `TargetDriftPreset` to the `Report` class and re-run the
        analysis and inspect the result. Have your targets drifted?

5. Evidently reports are meant for debugging, exploration and reporting of results. However, as we stated in the
    beginning, what we are actually interested in methods automatically detecting when we are beginning to drift. For
    this we will need to look at Test and TestSuites:

    1. Lets start with a simple test that checks if there are any missing values in our dataset:

        ```python
        from evidently.test_suite import TestSuite
        from evidently.tests import TestNumberOfMissingValues
        data_test = TestSuite(tests=[TestNumberOfMissingValues()])
        data_test.run(reference_data=reference, current_data=current)
        ```

        again we could run `data_test.save_html` to get a nice view of the results (feel free to try it out) but
        additionally we can also call `data_test.as_dict()` method that will give a dict with the test results.
        What dictionary key contains the if all tests have passed or not?

    2. Take a look at this [colab notebook](https://colab.research.google.com/drive/1p9bgJZDcr_NS5IKVNvlxzswn6er9-abl)
        that contains all tests implemented in Evidently. Pick 5 tests of your choice, where at least 1 fails by default
        and implement them as a `TestSuite`. Then try changing the arguments of the test so they better fit your
        usecase and get them all passing.

6. (Optional) When doing monitoring in practice, we are not always interested in running on all data collected from our
    API maybe only the last `N` entries or maybe just from the last hour of observations. Since we are already logging
    the timestamps of when our API is called we can use that for filtering. Implement a simple filter that either takes
    an integer `n` and returns the last `n` entries in our database or some datetime `t` that filters away observations
    earlier than this.

7. Evidently by default only supports structured data e.g. tabular data (so does nearly every other framework). Thus,
    the question then becomes how we can extend unstructured data such as images or text? The solution is to extract
    structured features from the data which we then can run the analysis on.

    1. (Optional) For images the simple solution would be to flatten the images and consider each pixel a feature,
        however this does not work in practice because changes in the individual pixels does not really tell anything
        about the image. Instead we should derive some feature such as:

        * Average brightness
        * Contrast of image
        * Image sharpness
        * ...

        These are all numbers that can make up a feature vector for a image. Try out doing this yourself, for example by
        extracting such features from MNIST and FashionMNIST datasets, and check if you can detect a drift between the two
        sets.

    2. (Optional) For text a common approach is to extra some higher level embedding such as the very classical
        [GLOVE](https://nlp.stanford.edu/projects/glove/) embedding. Try following
        [this tutorial](https://github.com/evidentlyai/evidently/blob/main/examples/how_to_questions/how_to_run_calculations_over_text_data.ipynb)
        to understand how drift detection is done on text.

    3. Let's instead take a deep learning based approach to doing this. Let's consider the
        [CLIP](https://arxiv.org/abs/2103.00020) model, which is normally used to do image captioning. For our purpose
        this is perfect because we can use the model to get abstract feature embeddings for both images and text:

        ```python
        from PIL import Image
        import requests
        # requires transformers package: pip install transformers
        from transformers import CLIPProcessor, CLIPModel

        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)

        # set either text=None or images=None when only the other is needed
        inputs = processor(
            text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True
        )

        img_features = model.get_image_features(inputs['pixel_values'])
        text_features = model.get_text_features(inputs['input_ids'], inputs['attention_mask'])
        ```

        Both `img_features` and `text_features` are in this case a `(512,)` abstract feature embedding, that should be
        able to tell us something about our data distribution. Try using this method to extract features on two
        different datasets like CIFAR10 and SVHN if you want to work with vision or
        [IMDB movie review](https://www.kaggle.com/code/lakshmi25npathi/sentiment-analysis-of-imdb-movie-reviews) and
        [Amazon review](https://www.kaggle.com/datasets/PromptCloudHQ/amazon-echo-dot-2-reviews-dataset) for text. After
        extracting the features try running some data distribution testing you just learned about.

8. (Optional) If we have multiple applications and want to run monitoring for each application we often want also the
    monitoring to be a deployed application (that only we can access). Implement a `/monitoring/` endpoint that does
    all the reporting we just went through such that you have two endpoints:

    ```bash
    http://127.0.0.1:8000/iris_infer/?sepal_length=1.0&sepal_width=1.0&petal_length=1.0&petal_width=1.0 # user endpoint
    http://127.0.0.1:8000/iris_monitoring/ # monitoring endpoint
    ```

    Our monitoring endpoint should return an HTML page either showing an Evidently report or test suit. Try implementing
    this endpoint. We have implemented a solution in this [file](exercise_files/iris_fastapi.py) if you need help with
    how to return an HTML page from a FastAPI application.

## Data drift in the cloud

In the next section we are going to look at how we can incorporate the data drifting in our cloud environment. In
particular, we are going to be looking at how we can deploy a monitoring application that will run on a schedule and
then report those statistics directly back into GCP for us to study.

### ❔ Exercises

In this set of exercises we are going to deploy a machine learning model for sentiment analysis trained on
[Google Play Store Reviews](https://www.kaggle.com/datasets/prakharrathi25/google-play-store-reviews). The models task
is to predict if a users review is positive, neutral or negative in sentiment. We are then going to deploy a monitoring
service that will check if the distribution of the reviews have drifted over time. This may be useful if we are seeing
a decrease in the number of positive reviews over time, which may indicate that our application is not performing as
expected.

We have already created downloaded the training data, created a training script and trained a model for you.
The training data and the trained model is available to download from the following
[Google Drive folder](https://drive.google.com/drive/folders/19rZSGk4A4O7kDqPQiomgV0TiZkRpZ1Rs?usp=sharing) which can
be quickly downloaded by running the following commands (which uses the [gdown](https://github.com/wkentaro/gdown)
Python package):

```bash
pip install gdown
gdown --folder https://drive.google.com/drive/folders/19rZSGk4A4O7kDqPQiomgV0TiZkRpZ1Rs?usp=sharing
```

And the training script can be seen below. You are free to retrain the model yourself, but it takes about 30 mins to
train using a GPU. Overall the model scores around 74% accuracy on a hold-out test set. We recommend that you scroll
through the files to get an understanding of what is going on.

??? example "Training script for sentiment analysis model"

    ```python linenums="1" title="sentiment_classifier.py"
    --8<-- "s8_monitoring/exercise_files/sentiment_classifier.py"
    ```

1. To begin with lets start by uploading the training data and model to a GCP bucket. Upload to a new GCP bucket
    called `gcp_monitoring_exercise` (or something similar). Upload the training data and the trained model to the
    bucket.

    ??? success "Solution"

        This can be done by running the following commands or manually uploading the files to the bucket using the
        GCP console.

        ```
        gsutil mb gs://gcp_monitoring_exercise
        gsutil cp reviews.csv gs://gcp_monitoring_exercise/reviews.csv
        gsutil cp bert_sentiment_model.pt gs://gcp_monitoring_exercise/bert_sentiment_model.pt
        ```

2. Next we need to create a FastAPI application that takes a review as input and returns the predicted sentiment of
    the review. We provide a starting point for the application in the file below, that should be able to run as is.

    ??? example "Starting point for sentiment analysis API"

        ```python linenums="1" title="sentiment_api_starter.py"
        --8<-- "s8_monitoring/exercise_files/sentiment_api_starter.py"
        ```

    1. Confirm that you can run the application by running the following command in the terminal

        ```bash
        uvicorn sentiment_api_starter:app --reload
        ```

        You need the model file saved in the same directory as the application to run the application. Write a small
        `client.py` script that calls the application with a review and prints the predicted sentiment.

        ??? success "Solution"

            ```python
            import requests

            url = "http://localhost:8000/predict"
            review = "This is a great app, I love it!"
            response = requests.post(url, json={"review": review})
            print(response.json())
            ```

    2. Next, we need to extend the application in two ways. First instead of loading the model from our local computer,
        it should load from the bucket we just uploaded the model to. Secondly, we need to save the request data and the
        predicted label to the cloud. Normally this would best be suited in a database, but we are going to just save
        to the same bucket as the model. We just need to make sure each request is saved under a unique name (e.g. the
        time and date of the request). Implement both of these functionalities in the application. To interact with
        GCP buckets in Python you should install the `google-cloud-storage` package if you have not already done so.

        ```bash
        pip install google-cloud-storage
        ```

        ??? success "Solution"

            ```python linenums="1" title="sentiment_api.py"
            --8<-- "s8_monitoring/exercise_files/sentiment_api.py"
            ```

    3. You should confirm that the application is working locally before moving on. You can do this by running the
        following command in the terminal

        ```bash
        uvicorn sentiment_api:app --reload
        ```

        And use the same `client.py` script as before to confirm that the application is working. You should also check
        that the data is saved to the bucket.

    4. Write a small Dockerfile that containerize the application

        ??? success "Solution"

            ```docker linenums="1" title="sentiment_api.dockerfilepy"
            --8<-- "s8_monitoring/exercise_files/sentiment_api.dockerfile"
            ```

            which can be built by running the following command

            ```bash
            docker build -f sentiment_api.dockerfile -t sentiment_api:latest .
            ```

    5. Deploy the container to cloud run and confirm that the application still runs as expected.

        ??? success "Solution"

            The following four commands should be able to deploy the application to GCP cloud run. Make sure to replace
            `<location>`, `<project-id>` and `<repo-name>` with the appropriate values.

            ```bash
            gcloud artifacts repositories create <repo-name> --repository-format=docker --location=<location>
            docker tag sentiment_api:latest <location>-docker.pkg.dev/<project-id>/<repo-name>/sentiment_api:latest
            docker push <location>-docker.pkg.dev/<project-id>/<repo-name>/sentiment_api:latest
            gcloud run deploy sentiment-api \
                --image <location>-docker.pkg.dev/<project-id>/<repo-name>/sentiment_api:latest \
                --region <region> --allow-unauthenticated
            ```

    6. Make sure that the application still works by trying to send a couple of requests to the deployed application and
        make sure that the request/response data is correctly saved to the bucket.

        ??? success "Solution"

            To get the url of the deployed service you can run the following command

            ```bash
            gcloud run services describe sentiment-api --format 'value(status.url)'
            ```

            which can the be used in the `client.py` script to call the deployed service.

3. We now have a working application that we are ready to monitor for data drift in real time. We therefore need to now
    write a FastAPI application that takes in the training data and the predicted data and run evidently to check if the
    data or the labels have drifted. Furthermore, we again provide a starting point for the application below.

    ```python linenums="1" title="sentiment_monitoring_starter.py"
    --8<-- "s8_monitoring/exercise_files/sentiment_monitoring_starter.py"
    ```

    Look over the script and make sure you know what kind of features we are going to monitor?

    ??? success "Solution"

        The provided starting script makes use of two presets from evidently:
        [TextOverviewPreset](https://docs.evidentlyai.com/presets/text-overview) and
        [TargetDriftPreset](https://docs.evidentlyai.com/presets/target-drift). The first preset extracts descriptive
        text statistics (like number of words, average word length etc.) and runs data drift detection on these and the
        second preset runs target drift detection on the predicted labels.

    1. The script misses one key function to work: `#!python fetch_latest_data(n: int)` that should fetch the latest `n`
        predictions. Implement this function in the script.

        ??? success "Solution"

            ```python linenums="1" title="sentiment_monitoring.py"
            --8<-- "s8_monitoring/exercise_files/sentiment_monitoring.py"
            ```

    2. Test out the script locally. This can be done by downloading a couple of the request/response data from the
        bucket and running the script on this data.

    3. Write a small dockerfile that containerize the monitoring application

        ??? success "Solution"

            ```docker linenums="1" title="sentiment_api.dockerfilepy"
            --8<-- "s8_monitoring/exercise_files/sentiment_api.dockerfile"
            ```

4. We are now finally, ready to test our services. Since we need to observe some long term behavior this part may take
    some time to run depending on how you have exactly configured your. Below we have implemented a client script that
    are meant to call our service.

    !!! example "Training script for sentiment analysis model"

        ```python linenums="1" title="sentiment_client.py"
        --8<-- "s8_monitoring/exercise_files/sentiment_client.py"
        ```

    1. What does the client script do?

        ??? success "Solution"

            The client script will iteratively call our deployed sentiment analysis service every `wait_time` seconds.
            In each iteration it does:

            * Randomly samples a review for a list of positive, neutral and negative reviews
            * Randomly add negative phrases to the review. Each review is added if a randomly uniform number is lower
                than probability `negative_probability=min(count / args.max_iterations, 1.0), meaning that it becomes
                more and more likely that the negative phrases are added as the number of iterations increases.
            * Sends the review to the sentiment analysis service and saves the response to a file.

    2. Run the client script for 100 iterations.

## 🧠 Knowledge check

1. What are some common causes of data drift in machine learning models running in production?

    ??? success "Solution"

        * Seasonal changes: Consumers of machine learning models may change their behavior with the seasons. For
            example, a model that predicts sales may need to be retrained for the holiday season.
        * User behavior change: Trends in user behavior (most likely due to social media) may change the data
            distribution. For example, a model that predicts user engagement may need to be retrained if users start
            using the platform differently.
        * External factors: Changes in the environment that the model is operating in may change. For example, a model
            that predicts traffic may need to be retrained if the city changes its road layout.
        * Sensor degradation: If the model is based on sensor data, the sensors may degrade over time leading to a
            change in the data distribution. For example, a model that predicts the temperature may need to be retrained
            if the temperature sensor starts to degrade and becomes less accurate.

2. How would you go about setting thresholds to meaningful detect data drift in a model?

    ??? success "Solution"

        Threshold are most commonly set by examining historical data and identifying the natural viriability in the
        feature distributions. Setting a threshold that is too low may lead to false positives (too many alerts) and
        a high threshold may lead to false negatives (not detecting drift when it is present). Because of this duality,
        thresholds are often set by examining the trade-off between these two types of errors, often in a very iterative
        process.

3. In what types of applications or industries is data drift a significant concern and why?

    ??? success "Solution"

        Data drift is of significant concerns in applications where data changes frequently and the model is expected to
        generalize to new data. This could be industries such as e-commerce, financial forecasting and healthcare. For
        example in e-commerce the products that are being sold may quickly change and the model should be able to adapt
        to these changes.

4. What are some strategies to proactively mitigate data drift?

    ??? success "Solution"

        Any strategy that focuses on creating a more generalizing model e.g. one that is less sensitive to the exact
        distribution of the data. This could be done by using more data, using data augmentation, using more complex
        models, using ensemble models etc.

That ends the module on detection of data drifting, data quality etc. If this has not already been made clear,
monitoring of machine learning applications is an extremely hard discipline because it is not a clear-cut when we
should actually respond to feature beginning to drift and when it is probably fine. That comes down to the individual
application what kind of rules that should be implemented. Additionally, the tools presented here are also in no way
complete and are especially limited in one way: they are only considering the marginal distribution of data. Every
analysis that we're done have been on the distribution per feature (the marginal distribution), however as the image
below show it is possible for data to have drifted to another distribution with the marginal being approximately the
same.

<figure markdown>
![Image](../figures/data_drift_marginals.png){width="500"}
</figure>

There are methods such as [Maximum Mean Discrepancy (MMD) tests](https://jmlr.org/papers/v13/gretton12a.html) that are
able to do testing on multivariate distributions, which you are free to dive into. In this course we will just always
recommend to consider multiple features when doing decision regarding your deployed applications.
