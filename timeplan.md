---
layout: default
title: Timeplan
nav_order: 2
---

# Timeplan

[Slides](../slides/Intro%20to%20the%20course.pdf){: .btn .btn-blue }

The course is organised into *exercise* (2/3 of the course) days and *project* days (1/3 of the course).

*Exercise* days start at 9:00 in the morning with an lecture (15-30 min) that will give some context about atleast one
of the topics of that day. Additionally, previous days exercises may shortly be touched upon. The remaining of the day
will be spend on solving exercises either individually or in small groups. For some people the exercises may be fast to
do and for others it will take the hole day. We will provide help throughout the day. We will try to answer questions
on slack but help with be priorities to students physically on campus.

*Project* days are intended for project work and you are therefore responsable for making an agreement with your group
when and where you are going to work. The first project days there will be a lecture at 9:00 with project information.
Other project days we may also start the day with an external lecture, which we highly recommend that you participate
in. During each project day we will have office hours where you can ask questions for the project.

Below is an overall timeplan for each day, including the presentation topic of the day and the frameworks that you will
be using in the exercises.

## Week 1

In the first week you will be introduced to a number of development practises for organising and developing code,
especially with a focus on making everything reproducible.

Date | Day       | Presentation topic                                                 | Frameworks                           | Format
-----|-----------|--------------------------------------------------------------------|--------------------------------------|-----------
2/1  | Monday    | [Deep learning software](../slides/Deep%20Learning%20software.pdf) | Terminal, Conda, IDE, Pytorch        | Exercises
3/1  | Tuesday   | [MLOps: what is it?](../slides/What%20is%20MLOps.pdf)              | Git, CookieCutter, Pep8, DVC         | Exercises
4/1  | Wednesday | [Reproducibility](../slides/Reproducibility.pdf)                   | Docker, Hydra                        | Exercises
5/1  | Thursday  | [Debugging](../slides/Debugging%20ML%20Code.pdf)                   | Debugger, Profiler, Wandb, Lightning | Exercises
6/1  | Friday    | [Pytorch ecosystem](../slides/Projects.pdf)                        | -                                    | Projects

## Week 2

The second week is about automatization and the cloud. Automatization will help use making sure that our code
does not break when we make changes to it. The cloud will help us scale up our applications and we learn how to use
different services to help develop a full machine learning pipeline.

Date | Day       | Presentation topic                                              | Frameworks                                        | Format
-----|-----------|-----------------------------------------------------------------|---------------------------------------------------|-----------
9/1  | Monday    | [Continuous Integration](../slides/Continues%20Integration.pdf) | Pytest, Github actions, Pre-commit, CML           | Exercises
10/1 | Tuesday   | [The Cloud](../slides/Cloud%20Intro.pdf)                        | GCP Engine, Bucket, Container registry, Vertex AI | Exercises
11/1 | Wednesday | [Deployment](../slides/Deployment.pdf)                          | FastAPI, Torchservce, GCP Functions, Run          | Exercises
12/1 | Thursday  | Guest lecture                                                   | -                                                 | Projects
13/1 | Friday    | Guest lecture                                                   | -                                                 | Projects

## Week 3

For the final week we look into advance topics such as monitoring and scaling of applications. Monitoring is especially
important for the longivity for the applications that we develop, that we actually can deploy them either
locally or in the cloud and that we have the tools to monitor how they behave over time. Scaling of applications is an
important topic if we ever want our applications to be used by many people at the same time.

Date | Day       | Presentation topic                                                | Frameworks                          | Format
-----|-----------|-------------------------------------------------------------------|-------------------------------------|----------
16/1 | Monday    | Monitoring (Guest lecture)                                        | Evidently AI, OpenTelemetry, Signoz | Exercises
17/1 | Tuesday   | [Scalable applications](../slides/Distributed%20applications.pdf) | Pytorch, Lightning                  | Exercises
18/1 | Wednesday | -                                                                 | -                                   | Projects
19/1 | Thursday  | -                                                                 | -                                   | Projects
20/1 | Friday    | -                                                                 | -                                   | Exam
