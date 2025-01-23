# Scaling applications

[Slides](../slides/ScalingApplications.pdf){ .md-button }

<div class="grid cards" markdown>

- ![](../figures/icons/pytorch.png){align=right : style="height:100px;width:100px"}

    Learn how to set up distributed data loading in your PyTorch application

    [:octicons-arrow-right-24: M29: Distributed Data Loading](data_loading.md)

- ![](../figures/icons/lightning.png){align=right : style="height:100px;width:100px"}

    Learn how to do distributed training in PyTorch using `pytorch-lightning`

    [:octicons-arrow-right-24: M30: Distributed Training](distributed_training.md)

- ![](../figures/icons/pytorch.png){align=right : style="height:100px;width:100px"}

    Learn how to do scalable inference in PyTorch

    [:octicons-arrow-right-24: M31: Scalable Inference](inference.md)

</div>

This module is all about scaling the applications that we are building. Here we are going to use a very narrow
definition of *scaling*, namely that we want our applications to run faster. However, one should note that in general
*scaling* is a much broader term. There are many different ways to scale your applications and we are going to look at
three of these related to different tasks in machine learning algorithms:

- Scaling data loading
- Scaling training
- Scaling inference

We are going to approach the term *scaling* from two different angles and both should result in your application
running faster. The first approach is levering multiple devices, such as using multiple CPU cores or parallelizing
training across multiple GPUs. The second approach is more analytical, where we are going to look at how we can
design smaller/faster model architectures that run faster.

It should be noted that this module is specific to working with PyTorch applications. In particular, we are going to see
how we can both improve base PyTorch code and how to utilize PyTorch Lightning which we introduced in module
[M14 on boilerplate](../s4_debugging_and_logging/boilerplate.md) to improve the scaling of our applications. If your
application is written using another framework we can guarantee that the same techniques in these modules transfer to
that framework, but may require you to seek out how to specifically do it.

If you manage to complete all the modules in this session, feel free to check out the *extra* module on scalable
[hyperparameter optimization](../s10_extra/hyperparameters.md).

!!! tip "Learning objectives"

    The learning objectives of this session are:

    * Understand how data loading during training can be parallelized and have experimented with it
    * Understand the different paradigms for distributed training and can run multi-GPU experiments using the
        framework `pytorch-lightning`
    * Knowledge of different ways, including quantization, pruning, architecture tuning, etc. to improve inference
        speed
