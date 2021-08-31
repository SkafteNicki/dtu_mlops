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

## Conda environment

You probably already have [conda](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) installed 
on your laptop, which is great. Conda is an environment manager that helps you make sure that the dependencies of
different projects does not cross-contaminate each other. 

### Exercise

1. Download and install conda. Make sure that your installation is working by writing `conda help` in a terminal 
   and it should show you the help message for conda.

2. Create a new conda environment for the remaining of the exercises using `conda create -n "my_environment"`

3. Which commando gives you a list of the packages installed in the current environment (HINT: check the
   `conda_cheatsheet.pdf` file). How do you easily export this list to a text file?
