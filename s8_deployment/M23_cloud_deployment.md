---
layout: default
title: M23 - Cloud deployment
parent: S8 - Deployment
nav_order: 2
---

# Cloud deployment
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

We are now returning to using the cloud. In module [M18](../s6_the_cloud/M18_using_the_cloud) you should have
gone through the steps of having your code in your github repository to automatically build into a docker
container, store that, store data and pull it all together to make a training run. After the training is
completed you should hopefully have a file stored in the cloud with your trained model weights.

Todays exercises will be about serving those model weights to an end user. We focus on two different ways
of deploying our model, `Google cloud functions` and `Google Vertex AI endpoints`.

## Cloud Functions

Cloud functions are the easiest way to get started with deployment because they are what is called *serverless*.
For serverless deployment we still need a server to do the actual workload, however the core concept is that **you** 
do you have to manage the server. Everything is magically taken care of behind the scene.

### Exercises

1. Go to the start page of `Cloud Functions`. Can be found in the sidebar on the homepage or you can just
   search for it. Activate the service if not already active.

2. Click the `Create Function` button which should take you to a screen like the image below. Give it a name,
   set the server region to somewhere close by and change the authetication policy to 
   `Allow unauthenticated invocations` so we can access it directly from a browser. Remember to note down the
   *URL* of the service somewhere.
   <p align="center">
     <img src="../figures/gcp_cloud_functions.png" width="500" title="hover text">
   </p>

3. On the next page, for `Runtime` pick the `Python 3.9` option. This will make the inline editor show both
   a `main.py` and `requirements.py` file. Look over them. Click the `Deploy` button in the lower left corner.

4. Afterwards you should see a green check mark beside your function meaning that it is deployed. Click the
   `Test function` button which will take you to the testing page.
   <p align="center">
     <img src="../figures/gcp_test_function.png" width="800" title="hover text">
   </p>

5. If you know what the application does, it should come as no surprise that does not require any input. We
   therefore just send an empty request by clicking the `Test The Function` button. Does the function return
   the output you expected? Wait for the logs to show up. What do they show?

   1. Click on the metrics tab. Identify what each panel is showing.

   2. Go to the trigger tab and go to the url for the application.

   3. Checkout the logs tab. You should see that your application have already been invoked multiple times.

6. Next, we are going to create an application that actually takes some input so we can try to send it requests.
   We provide a very simple `sklearn_cloud_function.py` script to get started.

   1. Figure out what the script does and run the script. This should create a file with trained model.

   2. Next create a storage bucket and upload the model file to the bucket. You can either do this through the
      webpage or run the following commands:
      ```
      gsutil mb gs://<bucket-name>  # mb stands for make bucket
      gsutil cp <file-name> gs://<bucket-name>  # cp stands for copy
      ``
      check that the file is in the bucket.
   
   3. Create a new cloud function with the same initial settings as the first one. Choose also the `Python 3.9`
      but this time change code to something that can actually use the model we just uploaded. Here is a code
      snippet to help you:
      ```python
      from google.cloud import storage
      client = storage.Client()
      bucket = client.get_bucket("dtumlops")
      blob = bucket.get_blob("model.pkl")
      pickle_in = blob.download_as_string()
      my_model = pickle.loads(pickle_in)


      def knn_classifier(request):
        """ Will do stuff """
        request_json = request.get_json()
        print(request_json)


      ```
      HINT: if you want to test locally you need to install the `google-cloud-storage` API:
      ```bash
      pip install google-cloud-storage
      ```


