# Timeplan

[Slides](../slides/IntroToTheCourse.pdf){ .md-button }

The course is organised into *exercise* (2/3 of the course) days and *project* days (1/3 of the course).

*Exercise* days start at 9:00 in the morning with an lecture (15-30 min) that will give some context about at least one
of the topics of that day. Additionally, previous days exercises may shortly be touched upon. The remaining of the day
will be spend on solving exercises either individually or in small groups. For some people the exercises may be fast to
do and for others it will take the whole day. We will provide help throughout the day. We will try to answer questions
on slack but help with be priorities to students physically on campus.

*Project* days are intended for project work and you are therefore responsible for making an agreement with your group
when and where you are going to work. The first project days there will be a lecture at 9:00 with project information.
Other project days we may also start the day with an external lecture, which we highly recommend that you participate
in. During each project day we will have office hours where you can ask questions for the project.

Below is an overall timeplan for each day, including the presentation topic of the day and the frameworks that you will
be using in the exercises.

!!! note

    Current dates listed below are for January 2024 version of the course. The lectures and recordings are currently
    from January 2023 version of the course. Please note that for January 2024, the first week starts on a Tuesday and
    ends on a Saturday.

Recodings (link to drive folder with mp4 files):

* [🎥2023 Lectures](https://drive.google.com/drive/folders/1j56XyHoPLjoIEmrVcV_9S1FBkXWZBK0w?usp=sharing)
* [🎥2024 Lectures](https://drive.google.com/drive/folders/1mgLlvfXUT9xdg9EZusgeWAmfpUDSwfL6?usp=sharing)

## Week 1

In the first week you will be introduced to a number of development practices for organizing and developing code,
especially with a focus on making everything reproducible.

Date | Day       | Presentation topic                                                 | Frameworks                           | Format
-----|-----------|--------------------------------------------------------------------|--------------------------------------|-----------
2/1  | Tuesday    | [Deep learning software📝](../slides/DeepLearningSoftware.pdf) | Terminal, Conda, IDE, Pytorch        | [Exercises](../s1_development_environment/README.md)
3/1  | Wednesday   | [MLOps: what is it?📝](../slides/IntroToMLOps.pdf)  | Git, CookieCutter, Pep8, DVC         | [Exercises](../s2_organisation_and_version_control/README.md)
4/1  | Thursday | [Reproducibility📝](../slides/ReproducibilityAndSoftware.pdf) | Docker, Hydra                        | [Exercises](../s3_reproducibility/README.md)
5/1  | Friday  | [Debugging📝](../slides/DebuggingML.pdf) | Debugger, Profiler, Wandb, Lightning | [Exercises](../s4_debugging_and_logging/README.md)
6/1  | Saturday    | [Project work📝](../slides/Projects.pdf) | -                                    | [Projects](projects.md)

## Week 2

The second week is about automatization and the cloud. Automatization will help use making sure that our code
does not break when we make changes to it. The cloud will help us scale up our applications and we learn how to use
different services to help develop a full machine learning pipeline.

Date | Day       | Presentation topic                                              | Frameworks                                        | Format
-----|-----------|-----------------------------------------------------------------|---------------------------------------------------|-----------
8/1  | Monday    | [Continuous Integration📝](../slides/ContinuousIntegration.pdf)| Pytest, Github actions, Pre-commit, CML           | [Exercises](../s5_continuous_integration/README.md)
9/1 | Tuesday    | [The Cloud📝](../slides/Cloud%20Intro.pdf)                    | GCP Engine, Bucket, Artifact registry, Vertex AI | [Exercises](../s6_the_cloud/README.md)
10/1 | Wednesday | [Deployment📝](../slides/Deployment.pdf)                      | FastAPI, Torchserve, GCP Functions, GCP Run          | [Exercises](../s7_deployment/README.md)
11/1 | Thursday  | No lecture                                                   | -                                                 | [Projects](projects.md)
12/1 | Friday    | No lecture                                               | -                                                 | [Projects](projects.md)

## Week 3

For the final week we look into advance topics such as monitoring and scaling of applications. Monitoring is especially
important for the longivity for the applications that we develop, that we can deploy them either
locally or in the cloud and that we have the tools to monitor how they behave over time. Scaling of applications is an
important topic if we ever want our applications to be used by many people at the same time.

Date | Day       | Presentation topic                                                | Frameworks                          | Format
-----|-----------|-------------------------------------------------------------------|-------------------------------------|----------
15/1 | Monday    | [Monitoring📝](../slides/Monitoring.pdf)                      | Evidently AI, Prometheus, GCP Monitoring |  [Exercises](../s8_monitoring/README.md)
16/1 | Tuesday   | [Scalable applications📝](../slides/ScalingApplications.pdf)   | Pytorch, Lightning                  | [Exercises](../s9_scalable_applications/README.md)
17/1 | Wednesday | -                                                                 | -                                   | [Projects](projects.md)
18/1 | Thursday  | -                                                                 | -                                   | [Projects](projects.md)
19/1 | Friday    | -                                                                 | -                                   | Exam
