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

2. Create a new conda environment for the remaining of the exercises using
   ```bash
   conda create -n "my_environment"
   ``` 
   We really recommend that you use multiple conda environment during the course to make sure you do not 
   mix dependencies between your exercises.

3. When you create an environment with `conda`, how do you specify which python version it should be using?

4. Which `conda` commando gives you a list of the packages installed in the 
   current environment (HINT: check the `conda_cheatsheet.pdf` file in the `exercise_files` folder). 

   1. How do you easily export this list to a text file? Do this, and make sure you export it to
   a file called `enviroment.yml`. 
   
   2. Finally, inspect the file to see what is in it.

   3. The `enviroment.yml` file you have created is one way to secure *reproducibility* between users, because
   anyone should be able to get an exact copy of you enviroment if they have your `enviroment.yml` file. 
   Try creating a new environment directly from you `enviroment.yml` file and check that the packages being
   installed exactly matches what you originally had.

6. Which `conda` commando gives you a list of all the environments that you have created?

4. As the introduction states, it is fairly safe to use `pip` inside `conda` today.
   What is the corresponding `pip` command that gives you a list of all `pip` installed packages? 
   and how to you export this to a file called `requirements.txt`? 
   (We will revisit requirement files at a later point)

5. If you look through the requirements that both `pip` and `conda` produces then you will see that it
   is often filled with a lot more packages than what you are actually using in your project e.g. what 
   you write import statements: `from package import module`. One way to come around this is to use the 
   package `pipreqs`, that will automatically scan your project and create a requirement file specific to that.
   Lets try it out:

   1. Install `pipreqs`:
      ```bash
      pip install pipreqs
      ```

   2. Either try out `pipreqs` on one of your own projects or try it out on some
      other online project. What does the file `pipreqs` produce look like compared
      to the files produces by either `pip` or `conda`.
