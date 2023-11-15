# Continuous Integration

[Slides](../slides/Continues%20Integration.pdf){ .md-button }

<p align="center">
  <img src="../figures/icons/pytest.png" width="130">
  <img src="../figures/icons/actions.png" width="130">
  <img src="../figures/icons/precommit.png" width="130">
  <img src="../figures/icons/dockerhub.png" width="130">
  <img src="../figures/icons/cml.png" width="130">
</p>

Continues integration is an sub discipline of the general field of *Continues X*. Continuous X is one of the core
elements of modern Devops, and by extension MLOps. Continuous X assumes that we have a (long) developer pipeline
(see image below) where we want to make some changes to our code e.g:

* Update our training data or data processing
* Update our model architecture
* Something else...

Basically any code change we will expect will have a influence on the final result. The problem with
doing changes to the start of our pipeline is that we want the change to propagate all the way through
to the end of the pipeline.

<figure markdown>
![Image](../figures/continuous_x.png){ width="1000" }
<figcaption>
<a href="https://faun.pub/most-popular-ci-cd-pipelines-and-tools-ccfdce429867"> Image credit </a>
</figcaption>
</figure>

This is where *continuous X* comes into play. The word *continuous* here refer to the fact that the
pipeline should *continuously* be updated as we make code changes. You can also choose to think of this
as *automatization* of processes. The *X* then covers that the process we need to go through to
automate steps in the pipeline, depends on where we are in the pipeline e.g. the tools needed to
do continuous integration is different from the tools need to do continuous delivery.

In this session we are going to focus on *continuous integration (CI)*. As indicated in the image above, CI usually
takes care of the first part of the developer pipeline that has to do with the code base, code building and code
testing. This is paramount to step in automatization as we would rather catch bugs in the beginning of our pipeline
than in the end.

* How to write unittests for our applications
* How to automate tests being run on code changes
* How to secure we do not commit code that does not follow our code standards
* How we can automate building of docker images
* How we can automate training of our machine learning pipeline

!!! tip "Learning objectives"

    The learning objectives of this session are:

    * Being able to write unittests that covers both data and model in your ML pipeline
    * Know how to implement CI using Github actions such that tests are automatically executed on code changes
    * Can use pre-commit to secure that code that are not up to standard does not get committed
    * Know how to implement CI for continues building of containers
    * Basic knowledge how machine learning processes can be implemented in a continues way
