---
layout: default
title: M9 - Docker
parent: S3 - Reproducibility
nav_order: 1
---

<img style="float: right;" src="../figures/icons/docker.png" width="130">

# Docker
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

{: .important }
> Core module

<p align="center">
  <img src="../figures/docker.png" width="400">
  <br>
  <a href="https://www.reddit.com/r/ProgrammerHumor/comments/cw58z7/it_works_on_my_machine/"> Image credit </a>
</p>

While the above picture may seem silly at first, it is actually pretty close to how [docker](https://www.docker.com/)
came to existence. A big part of creating a MLOps pipeline, is that you are able to **reproduce** it. Reproducibility
goes beyond versioning our code with `git` and using `conda` environment to keep track of our python installations.
To really get reproducibility we need to also capture also system level components like

* operating system
* software dependencies (other than python packages)

Docker provides this kind of system-level reproducibility by creating isolated programs dependencies. In addition to
docker providing reproducibility, one of the key features are also scaleability which is important when we later on
are going to discuss deployment. Because docker is system-level reproducible, it does not (conceptually) matter if
we try to start our program on a single machine or a 1000 machines at once.

## Docker overview

Docker has three main concepts: **docker file**, **docker image** and **docker container**:

<p align="center">
  <img src="../figures/docker_structure.png" width="800" >
</p>

* A **docker file** is a basic text document that contains all the commands a user could call on the command line to
  run an application. This includes installing dependencies, pulling data from online storage, setting up code and
  what commands that you want to run (e.g. `python train.py`)

* Running, or more correctly *building* a docker file will create a **docker image**. An image is a lightweight,
  standalone/containerized, executable package of software that includes everything (application code, libraries,
  tools, dependencies etc.) necessary to make an application run.

* Actually *running* an image will create a **docker container**. This means that the same image can be launched
  multiple times, creating multiple containers.

The exercises today will focus on how to construct the actual docker file, as this is the first step to constructing
your own container.

## Docker sharing
The hole point of using docker is that sharing applications becomes much easier. In general, we have two options

* After creating the `Dockerfile` we can simply commit it to github (its just a text file) and then ask other users
  to simple build the image by themselves.

* After building the image ourself, we can choose to upload it to a *image registry* such as
  [Docker Hub](https://hub.docker.com/) where other can get our image by simply running `docker pull`, making them
  able to instantaneous running it as a container, as shown in the figure below

 <p align="center">
   <img src="../figures/docker_share.png" width="1000" title="Credit to https://www.ravirajag.dev/blog/mlops-docker">
 </p>

## Exercises

In the following exercises we guide you how to build a docker file for your MNIST repository that will make the
training and prediction a self contained application. Please make sure that you somewhat understand each step and do
not just copy of the exercise. Also note that you probably need to execute the exercise from a elevated terminal e.g.
with administrative privilege.

The exercises today are only an introduction to docker and some of the steps are going to be unoptimized from a
production setting view. For example we often want to keep the size of docker image as small as possible, which we are
not focusing on for these exercises.

If you are using `VScode` then we recommend install the
[docker VScode extension](https://code.visualstudio.com/docs/containers/overview) for easy getting an overview of
which images have been build and which are running. Additionally the extension named *Dev Containers* may also be
beneficial for you to download.

1. Start by [installing docker](https://docs.docker.com/get-docker/). How much trouble that you need to go through
   depends on your operating system. For Windows and Mac we recommend they install *Docker desktop*, which comes with
   a graphical user interface (GUI) for quickly viewing docker images and docker containers currently build/in-use.
   Windows users that have not installed WSL yet are going to have to do it now (as docker need it as backend for
   starting virtual machines) but you do not need to install docker in WSL.

2. Try running the following to confirm that your installation is working:

   ```bash
   docker run hello-world
   ```

   which should give the message

   ```bash
   Hello from Docker!
   This message shows that your installation appears to be working correctly.
   ```

3. Next lets try to download a image from docker hub. Download the `busybox` image:

   ```bash
   docker pull busybox
   ```

   which is an very small (1-5Mb) containerized application that contains the
   most essential GNU fileutils, shellutils etc.

4. After pulling the image, write

   ```bash
   docker images
   ```

   which should show you all images that are available. You should see the
   `busybox` image that we just downloaded.

5. Lets try to run this image

   ```bash
   docker run busybox
   ```

   you will get that nothing happens! The reason for that is we did that not
   provide any commands to `docker run`. We essentially just ask it to start
   the `busybox` virtual machine, do nothing and then close it again. Now, try
   again this time with

   ```bash
   docker run busybox echo "hello from busybox"
   ```

   Note how fast this process is. In just a few seconds, Docker is able to
   start a virtual machine, execute a command and kill it afterwards.

6. Try running

   ```bash
   docker ps
   ```

   what does this command do? What if you add `-a` to the end?

7. If we wanted to run multiple commands within the virtual machine, we can
   start it in *interactive mode*

   ```bash
   docker run -it busybox
   ```

   this can be a great way to investigate what the filesystem of our virtual
   machine looks like.

8. As you may have already notice by now, each time we execute `docker run` we
   can still see small remnants of the containers using `docker ps -a`. These
   stray containers can end up take a lot of disk space. To remove them, use
   `docker rm` where you provide the container id that you want to delete

   ```bash
   docker rm <container_id>
   ```

9. Lets, now move on to trying to construct an docker file ourself for our
   MNIST project. Create a file called `trainer.dockerfile`. The intention is that we want to develop one dockerfile
   for running our training script and one for doing predictions.

10. Instead of starting from scratch we nearly always want to start from some base image. For this exercise we are
    going to start from a simple `python` image. Add the following to your `Dockerfile`

    ```docker
    # Base image
    FROM python:3.7-slim
    ```

11. Next we are going to install some essentials in our image. The essentials more or less consist of a python
    installation. These instructions may seem familiar if you are using linux:

    ```docker
    # install python
    RUN apt update && \
        apt install --no-install-recommends -y build-essential gcc && \
        apt clean && rm -rf /var/lib/apt/lists/*
    ```

12. The previous two steps are common for any docker application where you want to run python. All the remaining steps
    are application specific (to some degree):

    1. Lets copy over our application (the essential parts) from our computer to the container:

       ```docker
       COPY requirements.txt requirements.txt
       COPY setup.py setup.py
       COPY src/ src/
       COPY data/ data/
       ```

       Remember that we only want the essential parts to keep our docker image as small as possible. Why do we need each
       of these files/folders to run training in our docker container?

    2. Lets set the working directory in our container and add commands that install the dependencies:

       ```docker
       WORKDIR /
       RUN pip install -r requirements.txt --no-cache-dir
       ```

       the `--no-cache-dir` is quite important. Can you explain what it does and why it is important in relation to
       docker.

    3. Finally, we are going to name our training script as the *entrypoint* for our docker image. The *entrypoint* is
       the application that we want to run when the image is being executed:

       ```docker
       ENTRYPOINT ["python", "-u", "src/models/train_model.py"]
       ```

       the `"u"` here makes sure that any output from our script e.g. any `print(...)` statements gets redirected to our
       terminal. If not included you would need to use `docker logs` to inspect your run.

13. We are now ready to building our docker file into a docker image

    ```bash
    docker build -f trainer.dockerfile . -t trainer:latest
    ```

    please note here we are providing two extra arguments to `docker build`. The `-f train.dockerfile .` (the dot is
    important to remember) indicates which dockerfile that we want to run (except if you named it just `Dockerfile`) and
    the `-t trainer:latest` is the respective name and tag that we se afterwards when running `docker images` (see image
    below). Please note that building a docker image can take a couple of minutes.

    <p align="center">
      <img src="../figures/docker_output.PNG" width="1000" >
    </p>

14. Try running `docker images` and confirm that you get output similar to the one above. If you succeeds with this,
    then try running the docker image

    ```bash
    docker run --name experiment1 trainer:latest
    ```

    you should hopefully see your training starting. Please note that we can start as many containers that we want at
    the same time by giving them all different names using the `--name` tag.

15. Remember, if you ever are in doubt how files are organized inside a docker image you always have the option to start
    the image in interactive mode:

    ```bash
    docker run -it --entrypoint sh {image_name}:{image_name}
    ```

16. When your training has completed you will notice that any files that is created when running your training script is
    not present on your laptop (for example if your script is saving the trained model to file). This is because the
    files were created inside your container (which is its own little machine). To get the files you have two options:

    1. If you already have a completed run then you can use

       ```bash
       docker cp
       ```

       to copy the files between your container and laptop. For example to copy a file called `trained_model.pt` from a
       folder you would do:

       ```bash
       docker cp {container_name}:{dir_path}/{file_name} {local_dir_path}/{local_file_name}
       ```

       Try this out.

    2. A much more efficient strategy is to mount a volume that is shared between the host (your laptop) and the
       container. This can be done with the `-v` option for the `docker run` command. For example, if we want to
       automatically get the `trained_model.pt` file after running our training script we could simply execute the
       container as

       ```bash
       docker run --name {container_name} -v %cd%/models:/models/ trainer:latest
       ```

       this command mounts our local `models` folder as a corresponding `models` folder in the container. Any file save
       by the container to this folder will be synchronized back to our host machine. Try this out! Note if you have
       multiple files/folders that you want to mount (if in doubt about file organization in the container try to do
       the next exercise first). Also note that the `%cd%` need to change depending on your OS, see this
       [page](https://stackoverflow.com/questions/41485217/mount-current-directory-as-a-volume-in-docker-on-windows-10)
       for help.

17. With training done we also need to write an application for prediction. Create a new docker image called
    `predict.dockerfile`. This file should call your `src/models/predict_model.py` script instead. This image will
    need some trained model weights to work. Feel free to either includes these during the build process or mount them
    afterwards. When you When you created the file try to `build` and `run` it to confirm that it works. Hint: if you
    are passing in the model checkpoint and prediction data as arguments to your script, your `docker run` probably
    need to look something like

    ```bash
    docker run --name predict --rm \
        -v %cd%/trained_model.pt:/models/trained_model.pt \  # mount trained model file
        -v %cd%/data/example_images.npy:/example_images.npy \  # mount data we want to predict on
        predict:latest \
        ../../models/trained_model.pt \  # argument to script, path relative to script location in container
        ../../example_images.npy
    ```

18. (Optional, requires GPU support) By default a virtual machine created by docker only have access to your `cpu` and
    not your `gpu`. While you do not necessarily have a laptop with a GPU that supports training of neural network
    (e.g. one from Nvidia) it is beneficial that you understand how to construct a docker image that can take advantage
    of a GPU if you were to run this on a machine in the future that have a GPU (e.g. in the cloud). It does take a bit
    more work, but many of the steps will be similar to building a normal docker image.

    1. There are three prerequisites for working with Nvidia GPU accelerated docker containers. First you need to have
       the Docker Engine installed (already taken care of), have Nvidia GPU with updated GPU drivers and finally have
       the [Nvidia container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)
       installed. The last part you not likely have not installed and needs to do.

    2. To test that everything is working start by pulling a relevant Nvidia docker image. In my case this is
       the correct image:

       ```bash
       docker pull nvidia/cuda:11.0.3-base-ubuntu20.04
       ```

       but it may differ based on what cuda vision you have. You can find all the different offical Nvidia images
       [here](https://hub.docker.com/r/nvidia/cuda). After pulling the image, try running the `nvidia-smi` command
       inside a container based on the image you just pulled. It should look something like this:

       ```bash
       docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi
       ```

       and should show an image like below:
       <p align="center">
          <img src="../figures/nvidia_smi.PNG" width="600">
       </p>
       If it does not work, try redoing the steps.

    3. We should hopefully have a working setup now for running Nvidia accelerated docker containers. Next step is to
       get Pytorch inside of our container, such that our Pytorch implementation also correctly identify the GPU.
       Luckily for us Nvidia provides a set of docker images for GPU-optimized software for AI, HPC and visualizations
       through their [NGC Catalog](https://docs.nvidia.com/ngc/ngc-catalog-user-guide/index.html#what-is-nvidia-ngc).
       The containers that have to do with Pytorch can be seen
       [here](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/index.html). Try pulling the latest:

       ```bash
       docker pull nvcr.io/nvidia/pytorch:22.07-py3
       ```

       It may take some time, because the NGC images includes a lot of other software for optimizing Pytorch
       applications. It may be possible for you to find other images for running GPU accelerated applications that have
       a smaller memory footprint, but NGC are the recommend and supported way.

    4. Lets test that this container work:

       ```bash
       docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:22.07-py3
       ```

       this should run the container in interactive mode attached to your current terminal. Try opening `python` in
       the container and try writing:

       ```python
       import torch
       print(torch.cuda.is_available())
       ```

       which hopefully should return `True`.

    5. Finally, we need to incorporate all this into our already developed docker files for our application. This is
       also fairly easy as we just need to change our `FROM` statement in the beginning of our docker file:

       ```docker
       FROM python:3.7-slim
       ```

       change to

       ```docker
       FROM  nvcr.io/nvidia/pytorch:22.07-py3
       ```

       try doing this to one of your docker files, build the image and run the container. Remember to check that your
       application is using GPU by printing `torch.cuda.is_available()`.

The covers the absolute minimum you should know about docker to get a working image and container. That said, if you are
actively going to be using docker in the near future, one thing to consider is the image size. Even these simple images
that we have build still takes up GB in size. A number of optimizations steps can be taken to reduce the image size for
you or your end user. If you have time you can read [this article](https://devopscube.com/reduce-docker-image-size/) on
different approaches to reduce image size. Additionally, you can take a look at the
[dive-in extension](https://www.docker.com/blog/reduce-your-image-size-with-the-dive-in-docker-extension/) for docker
desktop that lets you explore in depth your docker images.
