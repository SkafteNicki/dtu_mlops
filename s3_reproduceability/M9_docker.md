---
layout: default
title: M9 - Docker
parent: S3 - Reproduceability
nav_order: 1
---

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

<p align="center">
  <img src="../figures/docker.png" width="500" title="hover text">
</p>

While the above picture may seem silly at first, it is actually pretty close to how [docker](https://www.docker.com/) came to existence. A big part of creating a MLOps pipeline, is that you are able to **reproduce** it. Reproduceability goes beyond versioning our code with `git` and using `conda` enviroment to keep track of our python installations. To really get reproduceability we need to also encapture also system level components like

* operating system
* software dependencies (other than python packages)

Docker provides this kind of system-level reproduceability by creating isolated programs dependencies. In addition to docker providing reproduceability, one of the key features are also scaleability which is important when we later on are going to discuss deployment. Because docker is system-level reproduceable, it does not (conceptually) matter if we try to start our program on a single machine or a 1000 machines at once.

## Docker overview

Docker has two key concepts: *container*  and *image*:

* A *container* refers

* A image* refers 

<p align="center">
  <img src="../figures/docker_architecture.png" width="800" title="hover text">
</p>





### Exercises

1. Start by [installing docker](https://docs.docker.com/get-docker/). How much trouble that you need to go through
   depends on your operating system.

  

