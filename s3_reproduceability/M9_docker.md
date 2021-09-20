---
layout: default
title: M9 - Docker
parent: S3 - Reproduceability
nav_order: 1
---

# Docker
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

<p align="center">
  <img src="../figures/docker.png" width="400" title="hover text">
</p>

While the above picture may seem silly at first, it is actually pretty close to how [docker](https://www.docker.com/) came to existence. A big part of creating a MLOps pipeline, is that you are able to **reproduce** it. Reproducibility goes beyond versioning our code with `git` and using `conda` enviroment to keep track of our python installations. To really get reproducibility we need to also capture also system level components like

* operating system
* software dependencies (other than python packages)

Docker provides this kind of system-level reproducibility by creating isolated programs dependencies. In addition to docker providing reproduceability, one of the key features are also scaleability which is important when we later on are going to discuss deployment. Because docker is system-level reproduceable, it does not (conceptually) matter if we try to start our program on a single machine or a 1000 machines at once.

## Docker overview

Docker has three main concepts: **docker file**, **docker image** and **docker container**:

<p align="center">
  <img src="../figures/docker_structure.png" width="800" title="hover text">
</p>

* A **docker file** is a basic text document that contains all the commands a user could call on the commandline to run an application. This includes installing dependencies, pulling data from online storage, setting up code and what commands that you want to run (e.g. `python train.py`)

* Running, or more correctly *building* a docker file will create a **docker image**. An image is a lightweight, standalone/containerized, executable package of software that includes everything (application code, libraries, tools, dependencies etc.) necessary to make an application run. 

* Actually *running* an image will create a **docker container**. This means that the same image can be launched multiple times, creating multiple containers.

The exercises today will focus on how to construct the actual docker file, as this is the first step to constructing your own container.

## Docker sharing
The hole point of using docker is that sharing applications becomes much easier. In general, we have two options

* After creating the `Dockerfile` we can simply commit it to github (its just a text file) and then ask other users
to simple build the image themself.

* After building the image ourself, we can choose to upload it to a *image registry* such as [Docker Hub](https://hub.docker.com/)
where other can get our image by simply running `docker pull`, making them able to instantinius running it as a container, as shown in the figure below

 <p align="center">
   <img src="../figures/docker_share.png" width="1000" title="Credit to https://www.ravirajag.dev/blog/mlops-docker">
 </p>


### Exercises

In the following exercises we guide you how to build a docker file for your mnist reposatory that will make the training a self contained
application. Please make sure that you somewhat understand each step and do not just copy of the exercise.

If you are using `VScode` then we recommend install the [docker VCcode extension](https://code.visualstudio.com/docs/containers/overview)
for easy getting an overview of which images have been build and which are running.


1. Start by [installing docker](https://docs.docker.com/get-docker/). How much trouble that you need to go through depends on your operating system.

2. After installing docker we will begin to construct a docker file for our MNIST project. Create a file called `trainer.dockerfile`. Then intention
   is that we want to develop one dockerfile for running our training script and one for doing predictions.

3. Instead of starting from scracts we nearly always want to start from some base image. For this exercise we are going
   to start from a simple `python` image. Add the following to your `Dockerfile`
   ```docker
   # Base image
   FROM python:3.7-slim
   ```

4. Next we are going install dependencies. Here we take care of python requirement that our package needs to run. Add the following
   ```docker
   COPY ./ /app  # this will copy everything (code, data ect) into the docker container in a folder called app
   WORKDIR /app  # set the app folder as the working dir
   RUN pip install -r requirements.txt --no-cache-dir  # install dependencies
   ```
   the `--no-cache-dir` is quite important. Can you explain what it does and why it is important in relation to docker.

5. Next we are going to expose our training command such that we can execute it when the image is running:
   ```docker
   CMD ['python3', '/app/src/models/train_model.py']
   ```
   In total you should therefore end up with a file looking like this:
   ```docker
   FROM python:3.8-slim

   COPY ./ /app
   WORKDIR /app
   RUN pip install -r requirements.txt --no-cache-dir

   CMD ['python3', '/app/src/models/train_model.py']
   ```

5. We are now ready to building our docker file into a docker image
   ```bash
   docker build -f train.dockerfile . -t trainer:latest
   ```
   please note here we are providing two extra arguments to `docker build`. The `-f train.dockerfile .` (the dot is important to remember) 
   indicates which dockerfile that we want to run (except if you named it just `Dockerfile`) and the `-t trainer:latest` is the respective 
   name and tag that we se afterwards when running `docker images`. Note that building a docker image can take a couple of minutes.

 <p align="center">
   <img src="../figures/docker_output.PNG" width="1000" title="hover text">
 </p>

6. Try running `docker image` and confirm that you get output similar to the one above. If you succeds with this, then try running the
   docker image
   ```bash
   docker run --name experiment1 trainer:latest
   ```
   you should hopefully see your training starting. Please note that we can start as many containers that we want at the same time by giving them all different names using the `--name` tag.

7. As the final exercise create a new docker image called `predict.dockerfile`. This file should ofcause use the `src/models/predict_model.py`
   script instead. Additionally you need to remember that you also need to include some model weights in your docker image. When you created
   the file try to `build` and `run` it to confirm that it works.

8. (Optional) If you want to take a look on the inside of our docker image you can install [dive](https://github.com/wagoodman/dive) that can be used to inspect what we (or someone else) put into a docker image before running it.

9. By default a virtual machine created by docker only have access to your `cpu` and not your `gpu`. While you do not nessesarily have a laptop with a GPU that supports training of neural network (e.g. one from Nvidia) it is beneficial that you understand how to contruct a docker image that can take advantage of a GPU if you were to run this on a machine in the future that have a GPU (e.g. in the cloud). 

The covers the absolute minimum you should know about docker to get a working image and container. That said, if you are actively going
to be using docker in the near future, one thing to consider is the image size. Even these simple images that we have build still takes
up GB in size. A number of optimizations steps can be taken to reduce the image size for you or your end user.
