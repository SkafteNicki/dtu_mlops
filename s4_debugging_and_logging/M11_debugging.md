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

Debugging is very hard to teach and is one of the skills that just comes with experience. That said, you should
familiarize yourself with the build-in [python debugger](https://docs.python.org/3/library/pdb.html) as it may come 
in handy during the course. 

<p align="center">
  <img src="../figures/debug.jpg" width="700" title="hover text">
</p>

To invoke the build in python debugger you can either:
* If you are using an editor, then you can insert inline breakpoints (in VS code this can be done by pressing F9)
  and then execute the script in debug mode (inline breakpoints can often be seen as small red dots to the left of
  your code). The editor should then offer some interface to allow you step through your code.

* Set a trace directly with the python debugger by calling
  ```python
  import pdb
  pdb.set_trace()
  ```
  anywhere you want to stop the code. Then you can use different commands (see the `python_debugger_cheatsheet.pdf`)
  to step through the code.

### Exercises

We here provide a script `vae_mnist_bugs.py` which contains a number of bugs to get it running. Start by going over
the script and try to understand what is going on. Hereafter, try to get it running by solving the bugs. The following 
bugs exist in the script:

* One device bug (will only show if running on gpu, but try to find it anyways)
* One shape bug 
* One math bug 
* One training bug

Some of the bugs prevents the script from even running, while some of them influences the training dynamics.
Try to find them all. We also provide a working version called `vae_mnist_working.py` (but please try to find
the bugs before looking at the script). Successfully debugging and running the script should produce three files: 
`orig_data.png`, `reconstructions.png`, `generated_samples.png`. 