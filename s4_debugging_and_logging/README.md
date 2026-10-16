# Debugging, Profiling, Logging and Boilerplate

[Slides](../slides/day4_debugging_ML_code.pdf){ .md-button }

<div class="grid cards" markdown>

- ![](../figures/icons/debugger.png){align=right : style="height:100px;width:100px"}

    Learn how to use the debugger in your editor to find bugs in your code.

    [:octicons-arrow-right-24: M12: Debugging](debugging.md)

- ![](../figures/icons/profiler.png){align=right : style="height:100px;width:100px"}

    Learn how to use a profiler to identify bottlenecks in your code and from those profiles optimize the runtime of
    your programs.

    [:octicons-arrow-right-24: M13: Profiling](profiling.md)

- ![](../figures/icons/w&b.png){align=right : style="height:100px;width:100px"}

    Learn how to systematically log experiments and hyperparameters to make your code reproducible.

    [:octicons-arrow-right-24: M14: Logging](logging.md)

- ![](../figures/icons/lightning.png){align=right : style="height:100px;width:100px"}

    Learn how to use the `pytorch-lightning` framework to minimize boilerplate code and structure deep learning models.

    [:octicons-arrow-right-24: M15: Boilerplate](boilerplate.md)

</div>

Today we are initially going to go over three different topics that are all fundamentally necessary for any data
scientist or DevOps engineer:

- Debugging
- Profiling
- Logging

All three topics can be characterized by something you are probably already familiar with. Since you started
programming, you have done debugging, since nobody can write perfect code on the first try. Similarly, while you have
not directly profiled your code, I bet that you at some point have had some very slow code and optimized it to run
faster. Identifying and improving are the fundamentals of profiling code. Finally, logging is a very broad term and
refers to any kind of output from your applications that helps you at a later point identify the "performance" of your
application.

However, while we expect you to already be familiar with these topics, we do not expect all of you to be experts, as
these topics are rarely focused on. Today we are going to introduce some best practices and tools to
help you overcome every one of these three important topics. As the final topic for today, we are going to learn about
how we can *minimize* boilerplate and focus on coding what matters for our project instead of all the boilerplate to get
it working.

!!! tip "Learning objectives"

    The learning objectives of this session are:

    * Understand the basics of debugging and how to use a debugger to find bugs in your code
    * Be able to use a profiler to identify bottlenecks in your code and from those profiles optimize the runtime of
        your programs
    * Be familiar with an experiment logging framework for tracking experiments and hyperparameters of your code to
        make it reproducible
    * Be able to use the `pytorch-lightning` framework to minimize boilerplate code and structure deep learning models
