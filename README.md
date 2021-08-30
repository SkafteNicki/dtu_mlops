---
layout: default
title: Homepage
nav_order: 1
permalink: /
---

<p align="center">
    <h1 align="center">Machine Learning Operations</h1>
    <p align="center">Reposatory for the DTU MLOps course containing lectures and exercises.</p>
    <p align="center"><strong><a href="https://skaftenicki.github.io/dtu_mlops/">Checkout the homepage!</a></strong></p>
</p>

<p align="center"> 
  <img src="figures/mlops.png" width="1000" title="hover text">
</p>

## Course details

* Course responsable
    * Postdoc Nicki Skafte Detlefsen, nsde@dtu.dk
    * Professor Søren Hauberg, sohau@dtu.dk
* 5 ECTS
* 3 week period of January 2022
* Master course
* Grade: Pass/not passed
* Type of assessment: weekly project updates + final oral examination/presentation
* Recommended prerequisites: [02456 (Deep Learning)](https://kurser.dtu.dk/course/2021-2022/02456) or experience
with the following:
    * General understanding of machine learning (datasets, probability, classifiers, overfitting ect.) and 
    basic knowledge about deep learning (backpropagation, convolutional neural network, auto-encoders ect.)
    * Coding in [Pytorch](https://pytorch.org/)

## MLOps: What is it?

A compound of “machine learning” and “operations”, refers to the practice for collaboration and communication 
between data scientists and operations professionals to help manage production ML (or deep learning) lifecycle.
The life cycle consist of a design, development and operations phase that are all equal important to get a
functional machine learning model.

Reading resources:
* https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning.
  Introduction blog post for those that have never heard about MLOps
* https://towardsdatascience.com/ml-ops-machine-learning-as-an-engineering-discipline-b86ca4874a3f. Great document
  from google about the different levels of MLOps
* https://ml-ops.org/content/mlops-principles. The principles of MLOps
* https://papers.nips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf. Great paper about the
technical "depth" in machine learning (can also be found in the litterature folder)

## Course structure

The course is divided into a number of *sessions*, where each session may contain multiple *modules*.
Each day the intention is that you complete a single session.

## Setup


Start by cloning or downloading this repository
```
git clone https://github.com/SkafteNicki/dtu_mlops
```
if you do not have git installed (yet) we will touch upon it in the course.
Additionally, you should join our slack channel which we use for communication:

https://join.slack.com/t/slack-ddr8461/shared_invite/zt-qzk7ho8z-1tBT_SkkkxtpgMU8x197pg


## Learning objectives

A student who has met the objectives of the course will be able to:

<p align="center">
<b>Demonstrate a broad knowledge of state-of-the-art frameworks for doing research within deep learning</b>
</p>
  
This includes:
* Organise code in a efficient way for easy maintainability and shareability (git, cookiecutter)
* Being able to debug, visualize and monitor multiple experiments to assess model performance (tensorflow, wandb)
* Cable of using online computing services to scale experiments (azure)
* Demonstrate knowledge about different distributed training paradigms (data parallelism, model parallelism, ect) 
with deep learning and how to apply them (pytorch-lightning)
* Understand the importance of reproducibility and how to make sure experiments are reproducible (hydra)
* Efficiently do crossvalidation and hyperparameter searched (optuna)
* Deploy deep learning models in the wild (torchserve, azure)
* Conduct a research project in collaboration with follow students using the frameworks teached in the course

<p align="center">
  <img src="https://miro.medium.com/proxy/1*KBobA-DaVtQ8Px6P_-tNqQ.jpeg" width="700" title="hover text">
</p>

(Image credit to [link](https://medium.com/nybles/understanding-machine-learning-through-memes-4580b67527bf))
