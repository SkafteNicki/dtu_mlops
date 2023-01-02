---
layout: default
title: Projects
nav_order: 12
permalink: projects
---

# Project work

[Slides](../slides/Projects.pdf){: .btn .btn-blue }

Approximately 1/3 of the course time is dedicated to doing project work. The projects will serve as the basis of your
exam. In the project, you will essentially re-apply everything that you learn throughout the course to a self chosen
project. The overall goals with the project is:

* To formulate a project within the provided guidelines
* Apply the material though in the course to the problem
* Present your findings

In the projects you are free to work on whatever problem that you want. That said, the only requirement that we have is
that you should incorporate one of the listed below frameworks from the Pytorch ecosystem into your project.

## The Pytorch Ecosystem

The [Pytorch ecosystem](https://pytorch.org/ecosystem/) is a great place for finding open-source frameworks that can
help you accelerate your own research. However, it is important to note that the ecosystem is not a complete list of
all the awesome packages that exist to extend the functionality of Pytorch. For the project work you will need to
choose between one of three such frameworks which will serve as the basis of your project. The three frameworks are:

* [PyTorch Image Models](https://github.com/rwightman/pytorch-image-models). PyTorch Image Models (also known as TIMM)
  is the absolutly most used computer vision package (maybe except for `torchvision`). It contains models, scripts and
  pre trained for a lot of state-of-the-art image models within computer vision.

* [Transformers](https://github.com/huggingface/transformers). The Transformers repository from the Huggingface group
  focuses on state-of-the-art Natural Language Processing (NLP). It provides many pre-trained model to perform tasks on
  texts such as classification, information extraction, question answering, summarization, translation, text generation,
  etc in 100+ languages. Its aim is to make cutting-edge NLP easier to use for everyone.

* [Pytorch-Geometric](https://github.com/rusty1s/pytorch_geometric). PyTorch Geometric (PyG) is a geometric deep
  learning. It consists of various methods for deep learning on graphs and other irregular structures, also known as
  geometric deep learning, from a variety of published papers.

## Exercises for project day 1

Today is also dedicated to doing project work. Remember that the focus of the project work is not to demonstrate that
you can work with the biggest and baddest deep learning model, but instead that you show that you can incorporate the
tools that are taught throughout the course in a meaningful way.

Also note that the project is not expected to be very large in scope. It may simply be that you want to train X model
on Y data. You will approximately be given 4 full days to work on the project. It is better that you start out with a
smaller project and then add along the way if you have time.

1. (Optional) Familiar yourself with each of the libraries. One way to do this is to find relevant tutorials on each
   project and try to figure out the code. Such tutorials will give you a rough idea how the API for each library looks
   like.

2. Form groups! The recommended group size is 4 persons, but we also accept 3 or 5 man groups. Try to find other people
   based on what framework that you would like to work with.

3. Brainstorm projects! Try to figure out exactly what you want to work with and especially how you are going to
   incorporate the frameworks (we are aware that you are not familiar with every framework yet) that you have chosen to
   work with into your project. The **Final exercise** for today is to formulate a project description (see bottom of
   this page).

4. When you formed groups and formulated a project you are allowed to start working on the actual code. I have included
   a to-do list at the bottom that somewhat summaries what we have done in the course until know. You are **NOT**
   expected to fulfill all bullet points on the to-do list today, as you will continue to work on the project in the
   following two weeks.

Final exercise for today is making a project description. Write around half to one page about:

* Overall goal of the project
* What framework are you going to use (PyTorch Image Models, Transformer, Pytorch-Geometrics)
* How to you intend to include the framework into your project
* What data are you going to run on (initially, may change)
* What deep learning models do you expect to use

The project description will serve as an guideline for us at the exam that you have somewhat reached the goals that you
set out to do. For inspiration you can take a look at the following two projects from the last year:

1. [Classification of tweets using Transformers](https://github.com/nielstiben/MLOPS-Project)
2. [Classification of scientific papers using PyG](https://github.com/eyhl/group5-pyg-dtu-mlops)

By the end of the day (17:00) you should upload your project description (in the `README.md` file belonging to your
project repository) + whatever you have done on the project until now to your github repository. When this you have
done this, on DTU Learn go to assignments and hand in (as a group) the link to your github repository.

We will briefly (before next Monday) look over your github repository and project description to check that everything
is fine. If we have any questions/concerns we will contact you.

## Project checklist

Please note that all the lists are *exhaustive* meaning that I do not expect you to have completed very
point on the checklist for the exam.

### Week 1

* [ ] Create a git repository
* [ ] Make sure that all team members have write access to the github repository
* [ ] Create a dedicated environment for you project to keep track of your packages (using conda)
* [ ] Create the initial file structure using cookiecutter
* [ ] Fill out the `make_dataset.py` file such that it downloads whatever data you need and
* [ ] Add a model file and a training script and get that running
* [ ] Remember to fill out the `requirements.txt` file with whatever dependencies that you are using
* [ ] Remember to comply with good coding practices (`pep8`) while doing the project
* [ ] Do a bit of code typing and remember to document essential parts of your code
* [ ] Setup version control for your data or part of your data
* [ ] Construct one or multiple docker files for your code
* [ ] Build the docker files locally and make sure they work as intended
* [ ] Write one or multiple configurations files for your experiments
* [ ] Used Hydra to load the configurations and manage your hyperparameters
* [ ] When you have something that works somewhat, remember at some point to to some profiling and see if
      you can optimize your code
* [ ] Use wandb to log training progress and other important metrics/artifacts in your code
* [ ] Use pytorch-lightning (if applicable) to reduce the amount of boilerplate in your code

### Week 2

* [ ] Write unit tests related to the data part of your code
* [ ] Write unit tests related to model construction
* [ ] Calculate the coverage.
* [ ] Get some continuous integration running on the github repository
* [ ] (optional) Create a new project on `gcp` and invite all group members to it
* [ ] Create a data storage on `gcp` for you data
* [ ] Create a trigger workflow for automatically building your docker images
* [ ] Get your model training on `gcp`
* [ ] Play around with distributed data loading
* [ ] (optional) Play around with distributed model training
* [ ] Play around with quantization and compilation for you trained models

### Week 3

* [ ] Deployed your model locally using TorchServe
* [ ] Checked how robust your model is towards data drifting
* [ ] Deployed your model using `gcp`
* [ ] Monitored the system of your deployed model
* [ ] Monitored the performance of your deployed model

### Additional

* [ ] Revisit your initial project description. Did the project turn out as you wanted?
* [ ] Make sure all group members have a understanding about all parts of the project
* [ ] Create a presentation explaining your project
* [ ] Uploaded all your code to github
* [ ] (extra) Implemented pre*commit hooks for your project repository
* [ ] (extra) Used Optuna to run hyperparameter optimization on your model

## Exam

The exam consist of a written and oral element, and both contributes to the overall evaluation if you should pass or
not pass the course.

For the written part of the exam we provide an template folder called
[reports](https://github.com/SkafteNicki/dtu_mlops/tree/main/reports). As the first task you should copy the folder and
all its content to your project repository. Then, you jobs is to fill out the `README.md` file which contains the report
template. The file itself contains instructions on how to fill it out and instructions on using the included `report.py`
file. You will hand-in the template by simple including it in your project repository. By midnight on the 20/1 we will
scrape it automatically, and changes after this point are therefore not registered.

For the oral part of the exam you will be given a time slot where you have to show up for 5-7 min and give a very short
demo of your project. What we are interested in seeing is essentially a live demo of your deployed application/project.
We will possibly also ask questions regarding the overall curriculum of the course. Importantly, you should have your
deployed application, the github repository with your project code, `W&B` account and your `gcp` account ready before
you enter the exam so we can quickly jump around. We will send out an the time slots during the last week.
s