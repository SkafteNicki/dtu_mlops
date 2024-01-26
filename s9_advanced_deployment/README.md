# 09. Advance Model Deployment

<p align="center">
  <img src="../figures/icons/streamlit.png" width="130">
  <img src="../figures/icons/onnx.png" width="130">
  <img src="../figures/icons/kubernetes.png" width="130">
</p>

This module is still under development.

The goal of this module is to go over some more advanced topics related to model deployment. The topics for deployments
we have covered so far are great to get started with, but they are not enough to cover all the use cases you might have.
For example, want if you wanted to have a simple web interface for interacting with your model? Or what if you wanted to
deploy your model to a smart phone or web browser? Or what if you wanted to deploy your model to a Kubernetes cluster to
really scale it up? These are all topics that we will cover in this module.

All the topics are considered optional and you can pick and choose which ones you want to go through. They should all be
completely independent of each other so you can do them in any order you want.

!!! tip "Learning Objectives"

    The learning objectives of this session are:

        * Understand how to create simple web interfaces for your models using `streamlit`
        * Understand how to convert your models using the ONNX format such that it can run on a variety of different
          platforms
        * Understand how to deploy your models to a Kubernetes cluster using `kubeflow`
