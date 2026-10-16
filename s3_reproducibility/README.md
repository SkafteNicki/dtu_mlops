# Reproducibility

[Slides](../slides/day3_reproducibility_and_software.pdf){ .md-button }

<div class="grid cards" markdown>

- ![](../figures/icons/docker.png){align=right : style="height:100px;width:100px"}

    Learn how to create reproducible computing environments using `docker` and how to use them to run your code.

    [:octicons-arrow-right-24: M10: Docker](docker.md)

- ![](../figures/icons/hydra.png){align=right : style="height:100px;width:100px"}

    Learn how to use `hydra` to manage configuration files and how to integrate them into your code.

    [:octicons-arrow-right-24: M11: Config Files](config_files.md)

</div>

Today is all about reproducibility - one of those concepts that everyone agrees is very important and something should
be done about, but the reality is that it is very hard to ensure complete reproducibility. The last sessions have already
touched a bit on how tools like `conda` and code organization can help make code more reproducible. Today we are going
all the way to ensure that our scripts and our computing environment are fully reproducible.

## Why does reproducibility matter
Reproducibility is closely related to the scientific method:

> Observe -> Question -> Hypotheses -> Experiment -> Conclude -> Result -> Observe -> ...

Not having reproducible experiments essentially breaks the cycle between doing experiments and making conclusions.
If experiments are not reproducible, then we do not expect that others will arrive at the same conclusion as ourselves.
As machine learning experiments are fundamentally the same as doing chemical experiments in a laboratory, we should be
equally careful in making sure our environments are reproducible (think of your laptop as your laboratory).

Secondly, if we focus on why reproducibility matters especially in machine learning, it is part of the bigger challenge
of making sure that machine learning is **trustworthy**.

<figure markdown>
![Image](../figures/trustworthy_ml.drawio.png){ width="600" }
<figcaption>
Many different aspects are needed if trustworthy machine learning is ever going to be a reality. We need robustness pipelines
so we can trust that they will not fail under heavy load. We need integrity to make sure that pipelines are
deployed if they are of high quality. We need explainability to make sure that we understand what our machine learning
models are doing, so it is not just a black box. We need reproducibility to make sure that the results of our models can
be reproduced over and over again. Finally, we need fairness to make sure that our models are not biased toward specific
populations. Figure inspired by this<a href="https://arxiv.org/abs/2209.06529"> paper</a>.
</figcaption>
</figure>

Trustworthy ML is the idea that machine learning agents *can* be trusted. Take the example of a machine
learning agent being responsible for medical diagnoses. It is very clear that we need to be able to trust that the
agent gives us the correct diagnosis for the system to work in practice. Reproducibility plays a big role here,
because without it we cannot be sure that the same agent deployed at two different hospitals will give the same
diagnosis (given the same input).

!!! tip "Learning objectives"

    The learning objectives of this session are:

    * Understand the importance of reproducibility in computer science
    * Be able to use `docker` to create a reproducible container, including how to build them from scratch
    * Understand different ways of configuring your code and how to use `hydra` to integrate with config files
