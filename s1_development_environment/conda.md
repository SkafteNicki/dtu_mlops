---
layout: default
title: M2 - Conda
parent: S1 - Development environment
nav_order: 2
---

<img style="float: right;" src="../figures/icons/conda.png" width="130">

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

{: .important }

> Core module

You probably already have [conda](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) installed on your laptop, which is great. Conda is an environment manager that helps you make sure that the dependencies of different projects do not cross-contaminate each other. However, one thing is having conda installed, another is to actually use it.

Before we get on with the exercises, it is probably important to mention the differences between `pip` and `conda`. Here is a great [summary](https://www.anaconda.com/blog/understanding-conda-and-pip), but it essentially boils down to this:

-   `pip` always installs python packages (in the form of python wheels and distributions) whereas `conda` can also install packages written in other languages because it installs from a binary file.
-   `pip` installs dependencies in a serialized-recursive way, meaning that it can lead to dependency issues because all other dependencies are ignored when we are installing the first, and so on. On the other hand, `conda` goes over all dependencies in the beginning, checking for compatibility before installing anything.
-   `pip` is bound to a specific python version, whereas `conda` can manage multiple python versions at the same time.

It is therefore highly recommended to use conda environments compared to python virtual environments. However, does that mean that you cannot mix `pip install` and `conda install`? If you are using `conda>=4.6`, then you should be fairly safe, because it has a built-in compatible layer. In general, what works for me is:

-   Use `conda` to create environments
-   Use `pip` to install packages in that environment (with a few exceptions like `pytorch`)

Finally, it is worth mentioning that `pip` and `conda` are not the only two environment managers that exist for Python.
[Pipenv](https://pypi.org/project/pipenv/) is another alternative that is often used that has its own pros and cons
compared to other environment managers.

## Exercises

1. Download and install `conda`. You are free to either install full `conda` or the much simpler version `miniconda`.
   The core difference between the two packages is that `conda` already comes with a lot of packages that you would
   normally have to install with `miniconda`. The downside is that `conda` is a much larger package which can be a
   huge disadvantage on smaller devices. Make sure that your installation is working by writing `conda help` in a
   terminal and it should show you the help message for conda. If this does not work you probably need to set some
   system variable to
   [point to the conda installation](https://stackoverflow.com/questions/44597662/conda-command-is-not-recognized-on-windows-10)

2. Create a new conda environment for the remaining of the exercises using

    ```bash
    conda create -n "my_environment"
    ```

    We recommend that you use multiple conda environments throughout the course to ensure that you do not mix dependencies between exercises.

3. When you create an environment with `conda`, how do you specify which python version to use?
4. Which `conda` command gives you a list of the packages installed in the current environment (HINT: check the `conda_cheatsheet.pdf` file in the `exercise_files` folder)?

    1. How do you easily export this list to a text file? Do this and make sure to export it to a file called `environment.txt`.
    2. Inspect the file to see what is in it.
    3. The `environment.txt` file you have created is one way to ensure reproducibility between users, because anyone should be able to get an exact copy of your environment if they have your `environment.txt` file. Try creating a new environment directly from your `environment.txt` file and ensure that the packages being installed exactly match what you originally had.

5. Which `conda` command gives you a list of all the environments you have created?

6. As the introduction states, it is generally safe to use `pip` inside `conda` today. What is the corresponding `pip` command that gives you a list of all `pip`-installed packages, and how do you export this to a file called `requirements.txt` (we will revisit requirements files later)?

7. If you look through the requirements that both `pip` and `conda` produce, you will often see that they are filled with many more packages than you are actually using in your project. What you are really interested in are the packages that you import in your code: `from package import module`. One way to address this is to use the `pipreqs` package, which will automatically scan your project and create a requirements file specific to it. Let's try it out:

    1. Install `pipreqs`:

        ```bash
        pip install pipreqs
        ```

    1. Either try out `pipreqs` on one of your own projects or try it out on some other online project.
       What does the `requirements.txt` file that `pipreqs` produces look like compared to the files produced
       by either `pip` or `conda`.
