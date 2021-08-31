---
layout: default
title: M2 - Conda
parent: S1 - Getting started
nav_order: 2
---

# Conda and virtual enviroments
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

You probably already have [conda](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) installed 
on your laptop, which is great. Conda is an environment manager that helps you make sure that the dependencies of
different projects does not cross-contaminate each other. However, one thing is having conda installed, another is to
actually use it. 

Before we get on with the exercises, it is probably important to mention the differences between `pip` and `conda`.
Here is a great [summary](https://www.anaconda.com/blog/understanding-conda-and-pip) but it essentially it boils down 
to this: 
* `pip` always install python packages (in the form of python wheels and distributions) whereas `conda` can
also install packages written in other languages because it installs from a binary file. 
* `pip` installs dependencies in a serialized-recursive way, meaning that it can lead to dependencies issues, because all other
dependencies are ignored when we are installing the first and so on. On the other hand, `conda` go over all dependencies in the
beginning checking for compatibility before installing anything.
* `pip` is bound to a specific python version, whereas `conda` can manage multiple python versions at the same time

It is therefore highly recommended to use conda enviroments compared to python virtual enviroments. However, does that mean
that you cannot mix `pip install` and `conda install`? If you are using `conda>=4.6` then you should be fairly safe, because
it has build in compatible layer. In general, what works for me
* Use `conda` to create enviroments
* Use `pip` to install packages in that enviroment (with a few exceptions like `pytorch`)

### Exercise

1. Download and install `conda`. You are also free to use `miniconda` if that rocks your boat. Make sure that 
   your installation is working by writing `conda help` in a terminal and it should show you the help message 
   for conda. If this does not work you probably need to set some system variable to 
   [point to the conda installation](https://stackoverflow.com/questions/44597662/conda-command-is-not-recognized-on-windows-10)

2. Create a new conda environment for the remaining of the exercises using `conda create -n "my_environment"`.
   We really recommend that you use multiple conda enviroments during the course to make sure you do not mix
   dependencies between your exercises.

3. Which `conda` commando gives you a list of the packages installed in the current environment (HINT: check the
   `conda_cheatsheet.pdf` file). How do you easily export this list to a text file?

4. Similar which `pip` commando give you a list of all `pip` installed packages? and how to you export this to
   a file called `requirements.txt`? (We will revisit requirement files at a later point)
