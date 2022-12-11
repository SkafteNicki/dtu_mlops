---
layout: default
title: M18 - Continuous containers
parent: S5 - Continuous Integration
nav_order: 4
---

<img style="float: right;" src="../figures/icons/dockerhub.png" width="130">

# Continuous docker building
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

The Github Actions we learned about in [M16](M16_github_actions.md) are an powerfull tool that can be used to much more
than simply running our tests tests that we write for our application. In this module we are going to look at how we can
use it for continuously building docker images. As you have already seen docker building can take a couple of minutes
to build each time we do changes to our codebase. For this reason we really just want to build a new image every time we
do a commit of our code. Thus, it should come as no surprise that we can also automatize the building process and
furthermore we can take advantage of online compute power to parallelize the process.

As discussed in the initial module on [docker](../s3_reproducibility/M9_docker.md),
[docker hub](https://hub.docker.com/) is an online solution for storing build docker images in the cloud that is then
easy to pull down on whatever machine you want to run on. Docker hub is
[free to use](https://www.docker.com/pricing/) for personal use, as long as the images you push are public. We are in
this session going to look how we can automatically build and push our docker builds to docker hub. In a
[future module](../s6_the_cloud/M21_using_the_cloud.md) we are also going to look at the exact same process of building
and pushing containers but this time to an general cloud provider.

## Exercises

For these exercises you can choose to work with any docker file of your choosing. If you want an easy docker file,
you can use the following:

```dockerfile
FROM busybox
CMD echo "Howdy cowboy"
```

Alternatively, you can choose to focus on automatizing the training and prediction docker files back from
[M9](../s3_reproducibility/M9_docker.md). You will most likely need to change the docker image for your applications
if they contains any references to your data e.g. you have an `COPY data/ data/` statement in the file. Since we do
not store our data in Github, we cannot copy it during the build process.

1. Start by pushing whatever docker file you want that should be continuously build to your repository

2. Start by creating a [Docker Hub account](https://hub.docker.com/)

3. Next, within Docker Hub create an access token by going to `Settings -> Security`. Click the `New Access Token`
   button and give it a name that you recognize.

4. Copy the newly created access token and head over to your Github repository online. Go to
   `Settings -> Secrets -> Actions` and click the `New repository secret`. Copy over the access token and give
   it the name `DOCKER_HUB_TOKEN`. Additionally, add two other secrets `DOCKER_HUB_USERNAME` and `DOCKER_HUB_REPOSITORY`
   that contains your docker username and docker repository name respectively.

5. Next we are going to construct the actual Github actions workflow file:

   ```yaml
   name: Docker Image CI

   on:
     push:
       branches: [ master ]

   jobs:
     build:
       runs-on: ubuntu-latest
       steps:
       - uses: actions/checkout@v2
       - name: Build the Docker image
         run: |
           echo "${{ secrets.DOCKER_HUB_TOKEN }}" | docker login \
             -u "${{ secrets.DOCKER_HUB_USERNAME }}" --password-stdin docker.io
           docker build . --file Dockerfile \
             --tag docker.io/${{ secrets.DOCKER_HUB_USERNAME }}/${{ secrets.DOCKER_HUB_REPOSITORY }}:$GITHUB_SHA
           docker push docker.io/${{ secrets.DOCKER_HUB_USERNAME }}/${{ secrets.DOCKER_HUB_REPOSITORY }}:$GITHUB_SHA
   ```

   The first part of the workflow file should look somewhat recognizable. However, the last three lines are where
   all the magic happens. Carefully go through them and figure out what they do. If you want some help you can looking
   at the help page for `docker login`, `docker build` and `docker push`.

6. Upload the workflow to your github repository and check that it is being executed. If everything you should be able
   to see the the build docker image in your container repository in docker hub.

7. Make sure that you can execute `docker pull` locally to pull down the image that you just continuesly build

8. (Optional) To test that the container works directly in github you can also try to include an additional
   step that actually runs the container.

   ```yaml
     - name: Run container
       run: |
         docker run ...
   ```

That ends the session on continues docker building. We are going to revisit this topic after introducing the basic
concepts of working in the cloud, as it will make our life easier in the long run when we get to continues deployment
(CD) that our containers are stored the same place where we are going to run them. For completeness it is worth
mentioning that docker hub also offers the possibility of building your images in a continues way, by specifying so
called [build rules](https://docs.docker.com/docker-hub/builds/).
