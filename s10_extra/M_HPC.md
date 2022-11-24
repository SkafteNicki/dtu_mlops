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
  <a href="https://hpc.uni.lu/old/blog/2019/luxembourg-meluxina-supercomputer-part-of-eurohpc/"> Image credit. </a>
</p>

### Exercises

The following exercises are focused on local students at DTU that want to use our local
[HPC resources](https://www.hpc.dtu.dk/). That said, the steps in the exercise are fairly general to other types
of cluster.

1. Start by accessing the cluster. This can either be through `ssh` in a terminal or if you want a graphical interface
   [thinlinc](https://www.cendio.com/thinlinc/download) can be installed. In general we recommend following the steps
   [here](https://www.hpc.dtu.dk/?page_id=2501) for DTU students as the setup depends on if you are on campus or not.

2. When you have access to the cluster we are going to start with the setup phase. In the setup phase we are going
   to setup the environment necessary for our computations. If you have accessed the cluster through graphical interface
   start by opening a terminal.

   1. The idea behind A HPC system often haStart by activating the modules we need `module avail` 
   
   2. Start by 


   3. Finally, the changes we have made are only active for our current terminal and will be forgotten when we close 
      it. To make the changes permanent we can include them in our `.bashrc` file.


When working 


