---
layout: default
title: M25 - Data drifting
parent: S8 - Monitoring
nav_order: 1
---

# Data drifting
{: .no_toc }

<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

---

Data drifting is one of the core reasons for model accuracy degrades over time in production. For machine learning
models, data drift is the change in model input data that leads to model performance degradation. In practical terms
this means that the model is receiving input that is outside of the scope that it was trained on, as seen in the figure
below. This shows that the underlying distribution of a particular feature has slowly been increasing in value over
two years

<p align="center">
  <img src="../figures/data_drift.png" width="700">
  <br>
  <a href="https://www.picsellia.com/post/what-is-data-drift-and-how-to-detect-it-with-mlops"> Image credit </a>
</p>

In some cases, it may be that if you normalize some feature in a better way that you are able to generalize your model
better, but this is not always the case. The reason for such a drift is commonly some external factor that you
essentially have no control over. That really only leaves you with one option: retrain your model on the newly received
input features and deploy that model to production. This process is probably going to repeat over the lifetime of your
application if you want to keep it up-to-date with the real world.

<p align="center">
  <img src="../figures/retrain_model.png" width="700">
  <br>
  <a href="https://www.evidentlyai.com/blog/machine-learning-monitoring-data-and-concept-drift"> Image credit </a>
</p>

We have now come up with a solution to the data drift problem, but there is one important detail that we have not taken
care of: When we should actually trigger the retraining? We do not want to wait around for our model performance to
degrade, thus we need tools that can detect when we are seeing a drift in our data.

## Exercises

For these exercises we are going to use the framework [Evidently](https://github.com/evidentlyai/evidently) developed by
[EvidentlyAI](https://www.evidentlyai.com/). Evidently currently supports both detection for both regression and
classification models. The exercises are in large taken from
[here](https://docs.evidentlyai.com/get-started/hello-world) and in general we recommend if you are in doubt about an
exercise to look at the [docs](https://docs.evidentlyai.com/) for API and examples.

Additionally, we want to stress that data drift detection, concept drift detection etc. is still an active field of
research and therefore exist multiple frameworks for doing this kind of detection. In addition to Evidently,
we can also mention [NannyML](https://github.com/NannyML/nannyml) and [WhyLogs](https://github.com/whylabs/whylogs).

1. Start by install Evidently

   ```python
   pip install evidently
   ```

   you will also need `scikit-learn` and `pandas` installed if you do not already have it.

2. Hopefully you already gone through session [S7 on deployment](../s7_deployment/S7.md). As part of the deployment to
   GCP functions you should have developed a application that can classify Iris data, based on a model trained by this
   [script](../s7_deployment/exercise_files/sklearn_cloud_functions.py). We are going to convert this into a FastAPI
   application for the purpose here:

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

      We have implemented a solution in this [file](exercise_files/iris_fastapi.py) if you need help.

   2. Next we are going to add some functionality to our application. We need to add that the input for the user is
      saved to a database whenever our application is called. However, to not slow down the response to our user we want
      to implement this as an *background task*. A background task is a function that should be executed after the user
      have got their response. Implement a background task that save the user input to a database implemented
      as a simple `.csv` file. You can read more about background tasks
      [here](https://fastapi.tiangolo.com/tutorial/background-tasks/). The header of the database should look something
      like this:

      ```csv
      time, sepal_length, sepal_width, petal_length, petal_width
      2022-12-28 17:24:34.045649, 1.0, 1.0, 1.0, 1.0
      2022-12-28 17:24:44.026432, 2.0, 2.0, 2.0, 2.0
      ...
      ```

   3. Call you API a number of times to generate some dummy data in the database.

4. Create a new `data_drift.py` file where we are going to implement the data drifting detection and reporting

7. (Optional) If we have multiple applications and want to run monitoring for each application we often want also the
   monitoring to be a deployed application (that only we can access). Implement a `/monitoring/` endpoint that does
   all the reporting we just went through such that you two endpoints:

   ```bash
   http://127.0.0.1:8000/iris_inference/?sepal_length=1.0&sepal_width=1.0&petal_length=1.0&petal_width=1.0 # user endpoint
   http://127.0.0.1:8000/iris_monitoring/ # monitoring endpoint
   ```

   as with our script the monitoring endpoint should return a `.pdf` with the results of the data analysis.

8. As an final exercise, we recommend that you try implementing this to run directly in the cloud. For this to work you
   will need to change the following:

   * Instead of saving the input to a local file you should either store it in GCP bucket or an
     [BigQuery](https://console.cloud.google.com/bigquery) SQL table (this is a better solution, but also out-of-scope
     for this course)


That ends the module on detection of data drifting.