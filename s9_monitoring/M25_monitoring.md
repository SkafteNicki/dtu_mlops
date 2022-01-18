---
layout: default
title: M25 - System monitoring
parent: S9 - Monitoring
nav_order: 1
---

# Monitoring
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

### Exercises

We are in this exercise going to look at how we can setup automatic alerting
such that we get an message every time one of our applications are not behaving
as expected.

1. Go to the `Monitoring` service. Then go to `Alerting` tab. 
   <p align="center">
     <img src="../figures/gcp_alert.png" width="800" title="hover text">
   </p>

2. Start by setting up an notification channel. A recommend setting up with an
   email.

3. Next lets create a policy. Clicking the `Add Condition` should bring up a
   window as below. You are free to setup the condition as you want but the
   image is one way bo setup an alert that will react to the number of times
   an cloud function is invoked (actually it measures the amount of log entries
   from cloud functions).
   <p align="center">
     <img src="../figures/gcp_alert_condition.png" width="800" title="hover text">
   </p>

3. After adding the condition, add the notification channel you created in one of
   the earlier steps. Remember to also add some documentation that should be send
   with the alert to better describe what the alert is actually doing.

4. When the alert is setup you need to trigger it. If you setup the condition as
   the image above you just need to invoke the cloud function many times. Here is
   a small code snippet that you can execute on your laptop to call a cloud function
   many time (you need to change the url and payload depending on your function): 
   ```python
   import time
   import requests
   url = 'https://us-central1-dtumlops-335110.cloudfunctions.net/function-2'
   payload = {'message': 'Hello, General Kenobi'}

   for _ in range(1000):
      r = requests.get(url, params=payload)
   ```

5. Make sure that you get the alert through the notification channel you setup.


### Open questions

Below are listed some technical hard problems regarding MLOps. These are meant
as inspiration to get you to deep dive more into using all the cloud services
that `gcp` offers. You are also free to continue work on your project.

* Currently testing takes place in Github, but it should come as no
  surprise that `gcp` can also take care of this. Implementing testing
  on `gcp`. This 
  [blogpost](https://mickeyabhi1999.medium.com/basic-ci-cd-on-google-cloud-platform-using-cloud-build-b5c33d6842a7)
  can probably help.

* In the lectures we setup cloud build to automatically build a docker
  container for training whenever we pushed code to our github repository.
  However, we also setup CI testing in github. If tests are failing on
  github the building of the docker image is still being done, essentially
  wasting our precious cloud credit. Setup a system so cloud building only
  commence when all tests are passing.

* Authenticating between `gcp`, `wandb` and `dvc` can be tricky to do in
  a secure way. Figure out how to use the Secret Manager in `gcp` to
  pass secrets e.g. API keys during the build process of docker images.
  This [page](https://docs.docker.com/develop/develop-images/build_enhancements/#new-docker-build-secret-information)
  may help

* We have already done deployment through `Cloud Functions`. The native extension
  to cloud functions is the service `Cloud Run` which allows for more than
  just code snippets to be deployed. Checkout this service and try to deploy
  a container using it.

* All deployments we have done in the course have been serverless, because
  it makes it easier for us to focus on the actual application we are trying
  to deploy instead of focusing on server management. That said, going through
  the trouble of using a server orchestrator yourself can be worth it in many
  situations. Figure out how to use kubernetes in `gcp`. It will involve getting
  familiar with the kubernetes API and probably also kubeflow for managing
  pipelines on the server.

* Vertex AI is the newest ML service on `gcp`. It combines many of the features
  of the AI platform service you have already used with the AutoML service. Figure
  out how to use Vertex AI service to either train a custom model or use their
  AutoML feature. This
  [blogpost](https://cloud.google.com/blog/topics/developers-practitioners/pytorch-google-cloud-how-train-and-tune-pytorch-models-vertex-ai)
  can be a good place to start.

* If you want different services to be able to talk to each other the correct way
  is to setup a system using [Pub and Sub](https://cloud.google.com/pubsub) 
  (publish and subscription) service in `gcp`. Essentially it allows a service
  to publish a message and other services to subscribe and react to it. For 
  example the AI platform could publish a messesage everytime a model was done
  training and cloud build could subscribe to that, automatically staring to
  build a docker image using the trained model. Investigate Pub and Sub and
  try to make two services talk to each other.

* In the deployment exercises you probably looked at least once on the logs. We can
  automate what we do with the logs using the Logs Explorer service, which collects 
  all logs from all services that you are using. Setup
  [Logs routing](https://cloud.google.com/logging/docs/routing/overview) for one of
  your deployed services to your cloud storage. Afterwards setup a VM that consumes
  the logs and accumulate them.
  
