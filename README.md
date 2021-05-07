![mlops](figures/mlops-loop-en.jpg)

# ????? Machine Learning Operations

This repository contains the exercises for the DTU course ????? Machine Learning Operations (MLOps). 
All exercises are writting in the [Python](https://www.python.org/) programming language and formatted 
into a combination of scripts and [Jupyter Notebooks](https://jupyter.org/). 

This repository borrows heavily from previous work, in particular:

* 

## MLOps: What is it?

A compound of “machine learning” and “operations”, refers to the practice for collaboration and communication 
between data scientists and operations professionals to help manage production ML (or deep learning) lifecycle.

Reading resourses:
* https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning.
  Introduction blog post for those that have never heard about MLOps
* https://towardsdatascience.com/ml-ops-machine-learning-as-an-engineering-discipline-b86ca4874a3f. Great document
  from google about the different levels of MLOps

## Setup

Start by cloning or downloading this reposatory
```
git clone https://github.com/SkafteNicki/02457_mlops
```
if you do not have git installed (yet) we will touch apon it in the course.


## Course details

* 5 ECTS
* 3 week period
* Master course
* Grade: Pass/not passed
* Type of assesment: weekly project updates + final oral examination/presentation
* Recommended prerequisites: [02456 (Deep Learning)](https://kurser.dtu.dk/course/2021-2022/02456) or experience
with the following:
    * General understanding of machine learning (datasets, probability, classifiers, overfitting ect.) and 
    basic knowledge about deep learning (backpropergation, convolutional neural network, auto-encoders ect.)
    * Coding in [Pytorch](https://pytorch.org/)

## 


### Week 1

The first week is all about getting set up for the following two weeks. In particular this week focus setting
up a good practise for how to organise and develop code.

|        |  Presentation topic                    | Framework/exercise
|--------|-----------------------------------|--------------------
|Monday  |  How autodiff changed the world   | Freshup on pytorch
|Tuesday |  Code organisation: why it matters | Conda + Github + docker
|Wednesday | Debugging and visualization | Tensorboard, wandb
|Thursday | Project overview: pytorch ecosystem | Project work
|Friday  |  - | Project work

### Week 2

The second week is about scalability. While many times it does not require huge resources to do development,
there are always certain phases that requires you to scale your experiments. In this week we will focus on 
getting conftable on how to write distributed application and how we can run them

|            |  Presentation topic  | Framework/exercise
|--------|-----------------------------------|--------------------
|Monday      |  Training in the sky | AWS, azura, google cloud
|Tuesday    |  Distributed training: a overview | Pytorch Lightning
|Wednesday | Reproducibility: configuration files | Hydra
|Thursday   | - | Project work
|Friday     |  - | Project work

### Week 3

The last week is about extentions, that both may benefit production settings and research settings.

|        |  Presentation topic  | Framework/exercise
|--------|-----------------------------------|--------------------
|Monday  |  Cross validation and hyperparameters | Optuna
|Tuesday |  Conputing on data you do not own | PySyft + opacus
|Wednesday | Deployment | Torchserve + bentoml
|Thursday | - | Project work
|Friday  |  - | Project presentations

## Learning objectives

A student who has met the objectives of the course will be able to:

* Demonstrate a broad knowledge of state-of-the-art frameworks for doing research within deep learning 
* Organise code in a efficient way for easy maintainability and shareability (git + docker)
* Being able to debug, visualize and monitor multiple experiments to assess model performance (wandb)
* Cable of using online computing services to scale experiments (AWS)
* Demonstrate knowledge about different distributed training paradigms (data parallelism, model parallelism, ect) 
with deep learning and how to apply them (pytorch-lightning)
* Understand the importance of reproducibility and how to make sure experiments are reproducible (hydra)
* Efficiently do cross-validation and hyperparameter searched (optuna)
* Knowledge about privacy computing (Pysyft, opacus)
* Deploy deep learning models in the wild (torchserve)
* Conduct a research project in collaboration with follow students using the frameworks teached in the course
