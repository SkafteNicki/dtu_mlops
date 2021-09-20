---
layout: default
title: M11 - Debugging
parent: S4 - Debugging, Profiling and Logging
nav_order: 1
---

# Debugging
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

Debugging is very hard to teach and is one of the skills that just comes with experience. That said, there are good and bad ways to debug a program. 
We are all probably familiar with just inserting `print(...)` statements everywhere in our code. It is easy and can many times help narrow down where 
the problem happens. That said, this is not a great way of debugging when dealing with a very large codebase. You should therefore familiarize yourself 
with the build-in [python debugger](https://docs.python.org/3/library/pdb.html) as it may come in handy during the course. 

<p align="center">
  <img src="../figures/debug.jpg" width="700" title="hover text">
</p>

To invoke the build in python debugger you can either:

* Set a trace directly with the python debugger by calling
  ```python
  import pdb
  pdb.set_trace()
  ```
  anywhere you want to stop the code. Then you can use different commands (see the `python_debugger_cheatsheet.pdf`)
  to step through the code.

* If you are using an editor, then you can insert inline breakpoints (in VS code this can be done by pressing `F9`) and then execute the script in debug mode (inline breakpoints can often be seen as small red dots to the left of your code). The editor should then offer some interface to allow you step through your code. Here is a guide to using the build in debugger [in VScode](https://code.visualstudio.com/docs/python/debugging#_basic-debugging).

* Additionally, if your program is stopping on an error and you automatically want to start the debugger where it happens, then you can simply launch the program like this from the terminal
  ```
  python -m pdb -c continue myscript.py
  ```

### Exercise

We here provide a script `vae_mnist_bugs.py` which contains a number of bugs to get it running. Start by going over the script and try to understand what is going on. Hereafter, try to get it running by solving the bugs. The following 
bugs exist in the script:

* One device bug (will only show if running on gpu, but try to find it anyways)
* One shape bug 
* One math bug 
* One training bug

Some of the bugs prevents the script from even running, while some of them influences the training dynamics. Try to find them all. We also provide a working version called `vae_mnist_working.py` (but please try to find the bugs before looking at the script). Successfully debugging and running the script should produce three files: 

* `orig_data.png` containing images from the standard MNIST training set
* `reconstructions.png` reconstructions from the model
* `generated_samples.png` samples from the model

Again, we cannot stress enough that the exercise is actually not about finding the bugs but **using a proper** debugger to find them.