# Debugging, Profiling, Logging and Boilerplate

[Slides](../slides/DebuggingML.pdf){ .md-button }

<p align="center">
  <img src="../figures/icons/debugger.png" width="130">
  <img src="../figures/icons/profiler.png" width="130">
  <img src="../figures/icons/w&b.png" width="130">
  <img src="../figures/icons/lightning.png" width="130">
</p>

Today we are initially going to go over three different topics that are all fundamentally necessary for any data
scientist or DevOps engineer:

* Debugging
* Profiling
* Logging

All three topics can be characterized by something you probably already is familiar with. Since you started programming,
you have done debugging as nobody can write perfect code in the first try. Similarly, while you have not directly
profiled your code, I bet that you at some point have had some very slow code and optimized it to run faster.
Identifying and improving is the fundamentals of profiling code. Finally, logging is a very broad term and basically
refers to any kind of output from your applications that help you at a later point identify the "performance" of
you application.

However, while we expect you to already be familiar with these topics, we do not expect all of you to be expects in
this as it is very rarely topics that are focused on. Today we are going to introduce some best practices and tools to
help you overcome each and everyone of these three important topics.

As the final topic for today we are going to learn about how we can *minimize* boilerplate and focus on coding what
actually matters for our project instead of all the boilerplate to get it working.

!!! tip "Learning objectives"

    The learning objectives of this session are:

    * Understand the basics of debugging and how to use a debugger to find bugs in your code
    * Can use a profiler to identify bottlenecks in your code and from those profiles optimize the runtime of your
        programs
    * Familiar with an experiment logging framework for tracking experiments and hyperparameters of your code to make it
        reproducible
    * Be able to use `pytorch-lightning` framework to minimize boilerplate code and structure deep learning models
