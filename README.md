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

## Course plan

### Week 1

The first week is all about getting set up for the following two weeks. In particular this week focus setting
up a good practise for how to organise and develop code.

Date | Day       |  Presentation topic                 | Framework/exercise
-----|-----------|-------------------------------------|--------------------------
4/6  | Friday    | How autodiff changed the world      | Freshup on pytorch
7/6  | Monday    | Code organisation: why it matters   | Conda + Github + docker
8/6  | Tuesday   | Debugging and visualization         | Tensorboard, wandb
9/6  | Wednesday | Project overview: pytorch ecosystem | Project work
10/6 | Thursday  | -                                   | Project work

### Week 2

The second week is about scalability. While many times it does not require huge resources to do development,
there are always certain phases that requires you to scale your experiments. In this week we will focus on 
getting conftable on how to write distributed application and how we can run them

Date | Day       | Presentation topic                   | Framework/exercise
-----|-----------|--------------------------------------|-------------------------
11/6 | Friday    | Training in the sky                  | AWS, azura, google cloud
14/6 | Monday    | Distributed training: a overview     | Pytorch Lightning
15/6 | Tuesday   | Reproducibility: configuration files | Hydra
16/6 | Wednesday | -                                    | Project work
17/6 | Thursday  | -                                    | Project work

### Week 3

The last week is about extentions, that both may benefit production settings and research settings.

Date | Day       | Presentation topic                   | Framework/exercise
-----|-----------|--------------------------------------|--------------------
18/6 | Friday    | Cross validation and hyperparameters | Optuna
21/6 | Monday    | Conputing on data you do not own     | PySyft + opacus
22/6 | Tuesday   | Deployment                           | Torchserve + bentoml
23/6 | Wednesday | -                                    | Project work
24/6 | Thursday  | -                                    | Project presentations

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
