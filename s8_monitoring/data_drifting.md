---
layout: default
title: M25 - Data drifting
parent: S8 - Monitoring
nav_order: 1
---

<img style="float: right;" src="../figures/icons/evidentlyai.png" width="130">

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
[EvidentlyAI](https://www.evidentlyai.com). Evidently currently supports both detection for both regression and
classification models. The exercises are in large taken from
[here](https://docs.evidentlyai.com/get-started/hello-world) and in general we recommend if you are in doubt about an
exercise to look at the [docs](https://docs.evidentlyai.com/) for API and examples (their documentation can be a bit
lacking sometimes, so you may also have to dive into the source code).

Additionally, we want to stress that data drift detection, concept drift detection etc. is still an active field of
research and therefore exist multiple frameworks for doing this kind of detection. In addition to Evidently,
we can also mention [NannyML](https://github.com/NannyML/nannyml), [WhyLogs](https://github.com/whylabs/whylogs) and
[deepcheck](https://github.com/deepchecks/deepchecks).

1. Start by install Evidently

   ```python
   pip install evidently
   ```

   you will also need `scikit-learn` and `pandas` installed if you do not already have it.

2. Hopefully you already gone through session [S7 on deployment](../s7_deployment/S7.md). As part of the deployment to
   GCP functions you should have developed a application that can classify the
   [iris dataset](https://archive.ics.uci.edu/ml/datasets/iris), based on a model trained by this
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

      We have implemented a solution in this [file](exercise_files/iris_fastapi.py) (called v1) if you need help.

   2. Next we are going to add some functionality to our application. We need to add that the input for the user is
      saved to a database whenever our application is called. However, to not slow down the response to our user we want
      to implement this as an *background task*. A background task is a function that should be executed after the user
      have got their response. Implement a background task that save the user input to a database implemented
      as a simple `.csv` file. You can read more about background tasks
      [here](https://fastapi.tiangolo.com/tutorial/background-tasks/). The header of the database should look something
      like this:

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
   reference_data = datasets.load_iris(as_frame='auto').frame
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

   Standardize the dataframes such that they have the same column names and drop the time column from the `current_data`
   dataframe.

4. We are now ready to generate some reports about data drifting:

   1. Try executing the following code:

      ```python
      from evidently.report import Report
      from evidently.metric_preset import DataDriftPreset
      report = Report(metrics=[DataDriftPreset()])
      report.run(reference_data=reference, current_data=current)
      report.save_html('report.html')
      ```

      and open the generated `.html` page. What does it say about your data? Have it drifted? Make sure to poke around
      to understand what the different plots are actually showing.

   2. Data drifting is not the only kind of reporting evidently can make. We can also get reports on the data quality.
      Try first adding a few `Nan` values to your reference data. Secondly, try changing the report to

      ```python
      from evidently.metric_preset import DataDriftPreset, DataQualityPreset
      report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
      ```

      and re-run the report. Checkout the newly generated report. Again go over the generated plots and make sure that
      it picked up on the missing values you just added.

   3. The final report present we will look at is the `TargetDriftPreset`. Target drift means that our model is
      over/under predicting certain classes e.g. or general terms the distribution of predicted values differs from the
      ground true distribution of targets. Try adding the `TargetDriftPreset` to the `Report` class and re-run the
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
      and implement them as a `TestSuite`. Then try changing the arguments of the test so they better fit your usecase
      and get them all passing.

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
      [this tutorial](https://github.com/evidentlyai/evidently/blob/main/examples/how_to_questions/how_to_run_drift_report_for_text_encoders.ipynb)
      to understand how drift detection is done on text.

   3. Lets instead take a deep learning based approach to doing this. Lets consider the
      [CLIP](https://openai.com/blog/clip/) model, which is normally used to do image captioning. For our purpose this
      is perfect because we can use the model to get abstract feature embeddings for both images and text:

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
      inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

      img_features = model.get_image_features(inputs['pixel_values'])
      text_features = model.get_text_features(inputs['input_ids'], inputs['attention_mask'])
      ```

      Both `img_features` and `text_features` are in this case a `(512,)` abstract feature embedding, that should be
      able to tell us something about our data distribution. Try using this method to extract features on two different
      datasets like CIFAR10 and SVHN if you want to work with vision or
      [IMDB movie review](https://www.kaggle.com/code/lakshmi25npathi/sentiment-analysis-of-imdb-movie-reviews) and
      [Amazon review](https://www.kaggle.com/datasets/PromptCloudHQ/amazon-echo-dot-2-reviews-dataset) for text. After
      extracting the features try running some of the data distribution testing you just learned about.

8. (Optional) If we have multiple applications and want to run monitoring for each application we often want also the
   monitoring to be a deployed application (that only we can access). Implement a `/monitoring/` endpoint that does
   all the reporting we just went through such that you have two endpoints:

   ```bash
   http://127.0.0.1:8000/iris_infer/?sepal_length=1.0&sepal_width=1.0&petal_length=1.0&petal_width=1.0 # user endpoint
   http://127.0.0.1:8000/iris_monitoring/ # monitoring endpoint
   ```

   Our monitoring endpoint should return a HTML page either showing an Evidently report or test suit. Try implementing
   this endpoint. We have implemented a solution in this [file](exercise_files/iris_fastapi.py) if you need help with
   how to return an HTML page from a FastAPI application.

9. As an final exercise, we recommend that you try implementing this to run directly in the cloud. You will need to
   implement this in a container e.g. GCP Run service because the data gathering from the endpoint should still be
   implemented as an background task. For this to work you will need to change the following:

   * Instead of saving the input to a local file you should either store it in GCP bucket or an
     [BigQuery](https://console.cloud.google.com/bigquery) SQL table (this is a better solution, but also out-of-scope
     for this course)
   * You can either run the data analysis locally by just pulling from cloud storage predictions and training data
     or alternatively you can deploy this as its own endpoint that can be invoked. For the latter option we recommend
     that this should require authentication.

That ends the module on detection of data drifting, data quality etc. If this has not already been made clear,
monitoring of machine learning applications is an extremely hard discipline because it is not a clear cut when we should
actually respond to feature beginning to drift and when it is probably fine. That comes down to the individual
application what kind of rules that should be implemented. Additionally, the tools presented here are also in no way
complete and are especially limited in one way: they are only considering the marginal distribution of data. Every
analysis that we done have been on the distribution per feature (the marginal distribution), however as the image below
show it is possible for data to have drifted to another distribution with the marginal being approximatively the same.

<p align="center">
  <img src="../figures/data_drift_marginals.png" width="500">
</p>

There are methods such as [Maximum Mean Discrepancy (MMD) tests](https://jmlr.csail.mit.edu/papers/v13/gretton12a.html)
that are able to do testing on multivariate distributions, which you are free to dive into. In this course we will just
always recommend to consider multiple features when doing decision regarding your deployed applications.
