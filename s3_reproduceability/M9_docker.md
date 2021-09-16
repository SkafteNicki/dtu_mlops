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



### Exercises

In the following exercises we guide you how to build a docker file for your mnist reposatory that will make the training a self contained
application. Please make sure that you somewhat understand each step and do not just copy of the exercise.

1. Start by [installing docker](https://docs.docker.com/get-docker/). How much trouble that you need to go through depends on your operating system.

2. After installing docker we will begin to construct a docker file for our MNIST project. Create a file called `Dockerfile`.

3. Instead of starting from scracts we nearly always want to start from some base image. For this exercise we are going
   to start from a simple `python` image. Add the following to your `Dockerfile`
   ```docker
   # Base image
   FROM python:3.7-slim
   ```

4. Next we are going install dependencies. Here we take care of python requirement that our package needs to run. Add the following
   ```docker
   # copy relevant files to the docker image
   COPY setup.py setup.py
   COPY requirements.txt requirements.txt
   COPY Makefile Makefile
   ```

COPY ./ /app
WORKDIR /app
RUN pip install -r requirements_inference.txt
