---
layout: default
title: M16 - Continuous containers
parent: S5 - Continuous Integration
nav_order: 4
---

# Continuous docker building

`cml` integrates well with github and can give a taste of what a continuous machine learning
pipeline feels like. However, to take our applications to the next level we are going to look 
at how we can automatize docker building. As you have already seen docker building can take 
a couple of minutes to build each time we do changes to our codebase. For this reason we 
really just want to build a new image every time we do a commit of our code. Thus,
it should come as no surprise that we can also automatize the building process.

### Exercises

We are for now going to focus on automatizing out training and prediction docker images 
from session 3. You will most likely need to change the docker image for your training 
application if it contains your data e.g. you have an `COPY data/ data/` statement in it. 
Since the github repository does not store our data, we cannot copy it during the build 
process. We are later going to see how we can pull data during the build process, but 
this requires us to configure a cloud API storage account which will be part of session 6.

1. Below is an example workflow file for building a docker image. Create one of building 
   your training and one for building your prediction docker files. You will need to 
   adjust the file slightly.
   ```yaml
   name: Create Docker Container

   on: [push]

   jobs:
     mlops-container:
       runs-on: ubuntu-latest
       defaults:
         run:
           working-directory: ./week_6_github_actions
       steps:
         - name: Checkout
           uses: actions/checkout@v2
           with:
             ref: ${{ github.ref }}
         - name: Build container
           run: |
             docker build --tag inference:latest .
   ```
   Explain why it is better to have the two builds in separated workflow files instead of 1.

2. Push the files to do your github repository and make sure the workflow succeeds.

3. (Optional) To test that the container works directly in github you can also try to include an additional
   step that actually runs the container.
   ```yaml
     - name: Run container
       run: |
         docker run ...
   ```

Thats ends the session on Continuous X. We are going to revisit this topic when we get to deployment, which
is the other common factor in classical continuous X e.g. CI/CD=continuous integration and continuous deployment.
