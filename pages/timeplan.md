# Time plan

[Slides](../slides/day1_introduction_to_the_course.pdf){ .md-button }

The course is organized into *exercise* (2/3 of the course) days and *project* days (1/3 of the course).

*Exercise* days start at 9:00 in the morning with a lecture (usually 30-45 min) that will give some context about at
least one of the topics of that day. Additionally, previous days exercises may shortly be touched upon. The remaining
of the day will be spent on solving exercises either individually or in small groups. For some people the exercises
may be fast to do and for others it will take the whole day. We will provide help throughout the day. We will try to
answer questions on Slack, but help will be prioritized to students physically on campus.

*Project* days are intended for project work, and you are therefore responsible for making an agreement with your group
when and where you are going to work. The first project days there will be a lecture at 9:00 with project information.
Other project days we may also start the day with an external lecture, which we highly recommend that you participate
in. During each project day we will have office hours for you to ask questions regarding the project.

Below is an overall time plan for each day, including the presentation topic of the day and the frameworks that you will
be using in the exercises.

* [🎥2026 Lectures](https://panopto.dtu.dk/Panopto/Pages/Sessions/List.aspx?folderID=51273d06-22f1-4cb0-bc87-b3c900e41fc7)
* [🎥2025 Lectures](https://panopto.dtu.dk/Panopto/Pages/Sessions/List.aspx?folderID=14eeb1b7-5c39-4547-b7c3-b25d007cecd1)
* [🎥2024 Lectures](https://drive.google.com/drive/folders/1mgLlvfXUT9xdg9EZusgeWAmfpUDSwfL6?usp=sharing)
* [🎥2023 Lectures](https://drive.google.com/drive/folders/1j56XyHoPLjoIEmrVcV_9S1FBkXWZBK0w?usp=sharing)

## Week 1

In the first week you will be introduced to a number of development practices for organizing and developing code,
especially with a focus on making everything reproducible.

Date    | Day       | Presentation topic                                                 | Frameworks                           | Format
--------|-----------|--------------------------------------------------------------------|--------------------------------------|-----------
5/1/26  | Monday    | [Deep learning software📝](../slides/DeepLearningSoftware.pdf)     | Terminal, Conda, IDE, PyTorch        | [Exercises](../s1_development_environment/README.md)
6/1/26  | Tuesday   | [MLOps: what is it?📝](../slides/day2_introduction_to_mlops.pdf)                 | Git, CookieCutter, Pep8, DVC         | [Exercises](../s2_organisation_and_version_control/README.md)
7/1/26  | Wednesday | [Reproducibility📝](../slides/day3_reproducibility_and_software.pdf)      | Docker, Hydra                        | [Exercises](../s3_reproducibility/README.md)
8/1/26  | Thursday  | [Debugging📝](../slides/DebuggingML.pdf)                           | Debugger, Profiler, Wandb, Lightning | [Exercises](../s4_debugging_and_logging/README.md)
9/1/26  | Friday    | [Project work📝](../slides/Projects.pdf)                           | -                                    | [Projects](projects.md)

## Week 2

The second week is about automatization and the cloud. Automatization will help use making sure that our code
does not break when we make changes to it. The cloud will help us scale up our applications and we learn how to use
different services to help develop a full machine learning pipeline.

Date    | Day       | Presentation topic                                             | Frameworks                                        | Format
--------|-----------|----------------------------------------------------------------|---------------------------------------------------|-----------
12/1/26 | Monday    | [Continuous Integration📝](../slides/ContinuousIntegration.pdf)| Pytest, GitHub actions, Pre-commit, CML           | [Exercises](../s5_continuous_integration/README.md)
13/1/26 | Tuesday   | [The Cloud📝](../slides/CloudIntro.pdf)                        | GCP Engine, Bucket, Artifact registry, Vertex AI  | [Exercises](../s6_the_cloud/README.md)
14/1/26 | Wednesday | [Deployment📝](../slides/Deployment.pdf)                       | FastAPI, Torchserve, GCP Functions, GCP Run       | [Exercises](../s7_deployment/README.md)
15/1/26 | Thursday  | External lecture                                               | -                                                 | [Projects](projects.md)
16/1/26 | Friday    | No lecture                                                     | -                                                 | [Projects](projects.md)

## Week 3

For the final week we look into advance topics such as monitoring and scaling of applications. Monitoring is especially
important for the longevity for the applications that we develop, that we can deploy them either
locally or in the cloud and that we have the tools to monitor how they behave over time. Scaling of applications is an
important topic if we ever want our applications to be used by many people at the same time.

Date    | Day       | Presentation topic                                           | Frameworks                               | Format
--------|-----------|--------------------------------------------------------------|------------------------------------------|----------
19/1/26 | Monday    | [Monitoring📝](../slides/Monitoring.pdf)                     | Evidently AI, Prometheus, GCP Monitoring | [Exercises](../s8_monitoring/README.md)
20/1/26 | Tuesday   | [Scalable applications📝](../slides/ScalingApplications.pdf) | PyTorch, Lightning                       | [Exercises](../s9_scalable_applications/README.md)
21/1/26 | Wednesday | Summary lecture                                              | -                                        | [Projects](projects.md)
22/1/26 | Thursday  | No lecture                                                   | -                                        | [Projects](projects.md)
23/1/26 | Friday    | No lecture                                                   | -                                        | [Projects](projects.md)
