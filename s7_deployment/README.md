# 08. Model deployment

[Slides](../slides/Deployment.pdf){ .md-button }

<p align="center">
  <img src="../figures/icons/fastapi.png" width="130">
  <img src="../figures/icons/pytorch.png" width="130">
  <img src="../figures/icons/functions.png" width="130">
  <img src="../figures/icons/run.png" width="130">
</p>

Lets say that you have spend 1000 GPU hours and trained the most awesome model that you want to share with the
world. One way to do this is of course to just place all your code in a github repository, upload a file with
the trained model weights to your favorite online storage (assuming it is too big for github to handle) and
ask people to just download your code and the weights to run the code by themselves. This is a fine approach in a small
research setting, but in production you need to be able to **deploy** the model to an environment that is fully
contained such that people can just execute without looking (too hard) at the code.

<figure markdown>
  ![Image](../figures/deployment.jpg){ width="600" }
  <figcaption> <a href="https://soliditydeveloper.com/deployments"> Image credit </a> </figcaption>
</figure>

In this session we try to look at methods specialized towards deployment of models on your local machine and
also how to deploy services in the cloud.

!!! tip "Learning objectives"

    The learning objectives of this session are:

    * Understand the basics of requests and APIs
    * Can create custom APIs using the framework `fastapi` and run it locally
    * Knowledge about serverless deployments and how to deploy custom APIs using both serverless functions and
      serverless containers
