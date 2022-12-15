---
layout: default
nav_exclude: true
---

# Exam sheet

This is the template for the exam. Please do not remove anything from the template, only add your answers.
For including images, add it to the figures subfolder and then in this template add:
```
![my_image](figures/<image>.<extension>)
```


## Group information

### 1.
> Enter the group number you signed up on <learn.inside.dtu.dk>

### 2.
> Enter the study number for each member in the group
> Example:
> *sXXXXXX, sXXXXXX, sXXXXXX*
> Answer:

### 3.
> What framework did you choose to work with and did it help you complete the project?
>
> **Answer length: 100-200 words.**
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.

## Coding environment

> In the following section we are interested in learning more about you local development environment.

### 4.

> Explain how you managed dependencies in your project? Explain the process a new team member would have to go through
> to get an exact copy of your environment.
>
> **Answer length: 100-200 words**

### 5.

> We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your
> code. Did you fill out every folder or only a subset?
>
> **Answer length: 100-200 words**
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*

### 6.

> Did you implement any rules for code quality and format? Additionally, explain with your own words why these concepts matters in larger projects.
>
> **Answer length: 50-100 words.**
>

## Version control

> In the following section we are interested in learning more about you local development environment.

### 7.

> How many tests did you implement?

### 8.

### 9.

### 10.

### 11.

### 12.

## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### 13.

> What is the total code coverage (in percentage) of your code?
If you code had an code coverage of 100% (or close to), would you still trust it to be error free? Explain you reasoning.
>
> **Answer length: 100-200 words.**
>
> Example:
> *The total code coverage of code is X%, which includes all our source code*
> *We are far from 100% coverage of our code and even if we were then...*

### 14.

> Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and pull request can help improve version control.
>
> **Answer length: 100-200 words.**
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in addition to the main branch. To merge code we ...*


### 15.

> Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version control of your data. If no, explain a case where it would be beneficial to have version control of your data.
>
> **Answer length: 100-200 words.**

### 16.

> Discuss you continues integration setup. What kind of CI are you running (unittesting, linting, etc.)? Do you test multiple operating systems, python version etc. Do you make use of caching? Feel free to insert a link to one of your github actions workflow.
>
> **Answer length: 200-300 words.**
>
> Example:
> *We have organised our CI into 3 separate files: one for doing ..., one for running ... testing and one for running ... . In particular for our ..., we used ...*
> *An example of a triggered workflow can be seen here:*
> *https://github.com/nielstiben/MLOPS-Project/actions/runs/1728347853*

### 17.

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### 18.

> How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would run a experiment.
>
> **Answer length: 50-100 words.**
>
> Example:
> *We used a simple argparser, that worked in the following way: python my_script.py --lr 1e-3 --batch_size 25*

### 19.

> Reproducibility of experiments are important. Related to the last question, how did you secure that no information is lost when running experiments and that your experiments are reproducible?
>
> **Answer length: 100-200 words.**

### 20.

### 21.

### 22.

### 23.

### 24.

### 25.

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### 26.

<img style="float: right;" src="figures/overview.png" width="200">

> Include a figure that describes the overall architecture of your system and what services that you make use of.
> You can take inspiration from the figure to the right. Additionally in your own words, explain the overall steps
> in figure.
>
> **Answer length: 200-400 words**
>
> Example:
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and puch to github, it auto triggers ... and ... . From there the diagram shows ...*

### 27.

> Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these
> challenges?
>
> **Answer length: 200-400 words.**
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*

### 28.

> State the individual contributions of each team member.
>
> **Answer length: 50-200 words.**
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
