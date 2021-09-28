---
layout: default
title: M20 - Distributed Training
parent: S7 - Scalable applications
nav_order: 2
---

# Distributed Training
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


# 7. Distributed training

Today is about distributed training. We will start simple by making use of `DistributedData` which
is kind of the old standard for doing distributed training, and from there move on to format our
code that will make it agnostic towards device type and distributed setting using 
[Pytorch Lightning](https://pytorch-lightning.readthedocs.io/en/latest/).

## GPU on Azure

The exercise during `06_training_in_the_sky` showed you how to train your models using Azure. However,
as a small starting exercise today you should make yourself familiar with how to run Pytorch code on
a GPU in Azure.

1. Start by creating a GPU compute instance. You do this the same way you created the CPU instance
   where you just choose the `Virtual machine type` to be `GPU` (just choose the cheapest available type).

2. Take a look at the `fashion_trainer.py` script which simply trains a convolutional neural network on the
   Fashion MNIST dataset. Make sure that the script also runs on GPU, by adding `.to(device)` appropriate
   places in the code.

3. Write a Azure run script (using `ScriptRunConfig`) to run the `fashion_trainer.py` file on Azure.

4. Make a run on Azure on both CPU and GPU (by setting the `compute_targets` variable of your workspace)
   and see if running on GPU decreased the training time.

## Setup 

Sadly, Azure does not support starting multi-gpu instances with the free credit they give out when
creating an account. We are therefore today going to be using the local gpu cluster at DTU compute
to run the experiments. The first exercise (Distributed data loading) is possible to do own your
own laptop (assuming that you have one with multiple cores) but the remaining exercises need to
be executed in multi-gpu compute environment.

1. Download and install [thinlinc](https://www.cendio.com/thinlinc/download) for getting access to
   DTUs linux terminals

2. Login to the server `thinlinc.compute.dtu.dk` using your standard DTU credentials

2. Open a terminal (the big black icon in the lower ) and type 
   ```
   ssh gpuser1@clustername
   ```
   where clustername can either be `hyperion`, `oceanus` or `clymene`. You will be asked to give
   a password that is provided in todays lecture.

4. On one of the clusters (the home directory is connected between all clusters) create a folder with
   your study name
   ```
   mkdir studynumber
   ```
   you should only be working in this directory for all the exercises.

5. I have already created a conda environment that is ready to use
   ```
   conda activate dtu_mlops
   ```
   if you need to install additional packages, please take a copy of this environment and install in that
   ```
   conda create --name my_study_number --clone dtu_mlops
   ```

6. In addition, we also need to load the correct cuda package. This can be done by:
   ```
   module load CUDA/10.2
   ```

6. When you are ready to run a script, start by checking which GPUs are available:
   ```
   nvidia-smi
   ```
   Hereafter, to run your script simply do
   ```
   CUDA_VISIBLE_DEVICES=x python my_script.py
   ```
   where `x=0` if you want to run on GPU 0 or `x=1,2` if you want to run on the GPU 1 and 2.

## Distributed data loading

One common bottleneck in training deep learning models is the data loading (you maybe already saw this
during the profiling exercises). The reason


## Distributed Data

For this exercise we will briefly touch upon how to implement data parallel training in Pytorch using
their [nn.DataParallel](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html) class.

### Exercises

1. Create a small script where you take a copy of model `FashionCNN` from the `fashion_mnist.py` script.
   Instantiate the model and wrap `torch.nn.DataParallel` around it such that it can be executed in data
   parallel.

2. Try to run inference in parallel on multiple devices (pass a batch multiple times and time it). 
   Does data parallel decrease the inference time? If no, can you explain why that may be? Try playing
   around with the batch size, and see if data parallel is more beneficial for larger batch sizes.

## Going all the way

Moving beyond the just adjusting the number of workers in the dataloader or wrapping your model in
`torch.nn.DataParallel` is not so simple. To get your script to work with 
[Distributed data parallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel) would require more than just wrapping your model in the appropriate class. This is where training
frameworks such as *Pytorch Lightning* comes into play. As long as we format our model to the required
format of the framework we can enable distributed training with a single change of code.

### Exercises

