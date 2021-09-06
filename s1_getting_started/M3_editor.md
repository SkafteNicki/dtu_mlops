---
layout: default
title: M4 - Editor
parent: S1 - Getting started
nav_order: 3
---

# Editor/IDE
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
Notebooks can be great for testing out ideas, developing simple code and explaining and visualizing certain aspects
of a codebase. Remember that [Jupyter notebook](https://jupyter.org/) was created with intention to "...allows you 
to create and share documents that contain live code, equations, visualizations and narrative text." However, 
any larger machine learning project will require you to work in multiple `.py` files and here notebooks will provide 
a suboptimal workflow. Therefore, to for truly getting "work done" you will need a good editor / IDE. 

Many opinions exist on this matter, but for simplicity we recommend getting started with one of the following 3:

Editor		         | Webpage  	                  			| Comment (Biased opinion)
-------------------|------------------------------------|----------------------------------------------------------------------
Spyder             | https://www.spyder-ide.org/        | Matlab like environment that is easy to get started with
Visual studio code | https://code.visualstudio.com/     | Support for multiple languages with fairly easy setup
PyCharm            | https://www.jetbrains.com/pycharm/ | IDE for python professionals. Will take a bit of time getting used to

We highly recommend Visual studio (vs) code  if you do not already have a editor installed (or just want to try something new.). We therefore
put additional effort into explaining vs code.

Below you see an overview of the vs code interface

<p align="center">
  <img src="../figures/vscode.PNG" width="700" title="hover text">
</p>

The main components of vs code are:
* The action bar: Here you can 


### Exercise

1. Download and install one of the editors / IDE and make yourself familiar with it e.g. try out the editor
   on the files that you created in the final exercise in the last lecture.

The remaining of the exercises are specific to Visual studio code but we recommend that you try to answer the questions
if using another editor. In the `exercise_files` folder belonging to this session we have put cheat sheets for vs code
(one for windows and one for mac/linux), that can give you an easy overview of the different macros in vs code.

2. VS code is a general editor for many languages and to get proper *python* support we need to install some
extensions. In the `action bar` go to the `extension` tap and search for `python` in the marketplace. For here
we highly recommend installing the following packages:
   * `Python`: general python support
   * `Python for VSCode`: python syntax support
   * `Python Test Explorer for Visual Studio Code`: support for testing of python code (we get to that in a later lecture)
   * `Jupyter`: support for jupyter notebooks directly in VSCode

3. One of the most usefull tools with an editor is the ability to navigate a hole project using the build-in
`Explorer`. 

4. If you install the `Python` package you should see something like this in your status bar:





