---
layout: default
title: M34 - High Performance Clusters
parent: S10 - Extra
nav_order: 6
---

{: .warning }
> Module is still under development

# High Performance Clusters

As discussed in [the intro session on the cloud](../s6_the_cloud/S6.md), cloud providers offers near infinite
compute resources. However, using these resources comes at a hefty price often and it is therefore important to be
aware of another resource many have access to: High Performance Clusters or HPC. HPCs exist all over the world, and
many time you already have access to one or can easily get access to one. If you are an university student you most
likely have a local HPC that you can access through your institution. Else, there exist public HPC resources that
everybody (with a project) can apply for. As an example in the EU we have 
[EuroHPC](https://eurohpc-ju.europa.eu/index_en) initiative that currently has 8 different supercomputers with a
centralized location for [applying for resources](https://pracecalls.eu/) that are both open for research projects
and start-ups.

<p align="center">
  <img src="../figures/meluxina_overview.png" width="800">
  <br>
  Overview of the Meluxina supercomputer thats part of EuroHPC.
  <br>
  <a href="https://hpc.uni.lu/old/blog/2019/luxembourg-meluxina-supercomputer-part-of-eurohpc/"> Image credit </a>
</p>

Regardless which cluster you can get access to, in most cases it looks something like the image above, namely it is 
organized into different modules. When login to the cluster you will meet the front end of the cluster which contains
all the software needed to run computations. When you submit a job it will get sent to the backend modules which in most 
cases includes: general compute modules (CPU), acceleration modules (GPU), a memory module (RAM) and finally a storage 
module (HDD). Depending on your application you may need one module more than another. For example in deep learning 
the acceleration module is important but in physics simulation the general compute module / storage model is probably 
more important.

### Exercises

The following exercises are focused on local students at DTU that want to use our local
[HPC resources](https://www.hpc.dtu.dk/). That said, the steps in the exercise are fairly general to other types
of cluster. For the purpose of this exercise we are going to see how we can run this
[image classifier script](exercise_files/image_classifier.py), but feel free to work with whatever application you
want to.

1. Start by accessing the cluster. This can either be through `ssh` in a terminal or if you want a graphical interface
   [thinlinc](https://www.cendio.com/thinlinc/download) can be installed. In general we recommend following the steps
   [here](https://www.hpc.dtu.dk/?page_id=2501) for DTU students as the setup depends on if you are on campus or not.

2. When you have access to the cluster we are going to start with the setup phase. In the setup phase we are going
   to setup the environment necessary for our computations. If you have accessed the cluster through graphical interface
   start by opening a terminal.

   1. Lets start by setting up conda for controlling our dependencies. If you have not already worked with `conda`,
      please checkout module [M2 on conda](../s1_development_environment/M2_conda.md). In general you should be able to
      setup (mini)conda through these two commands:
      ```bash
      wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
      sh Miniconda3-latest-Linux-x86_64.sh
      ```

   2. Close the terminal and open a new for the installation to complete. Type `conda` in the terminal to check that
      everything is fine. Go ahead and create a new environment that we can install dependencies in
      ```bash
      conda create -n "hpc_env" python=3.10 --no-default-packages
      ```
      and activate it.

   3. Next, install all the requirements you need. If you want to run the image classifier script you can run this
      command in the terminal
      ```
      pip install -r image_classifier_requirements.txt
      ```
      using this [requirements file](exercise_files/image_classifier_requirements.txt).

3. Thats all the setup needed. You would need to go through the creating of environment and installation of requirements
   whenever you start a new project (no need for reinstalling conda). For the next step we need to look at how to submit
   jobs on the cluster.

   ```bash
   bsub
   qsub
   classstat
   qstat
   ```