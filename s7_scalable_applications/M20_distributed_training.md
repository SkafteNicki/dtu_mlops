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

In this module we are going to look at distributed training. Di



Today is about distributed training. We will start simple by making use of `DistributedData` which
is kind of the old standard for doing distributed training, and from there move on to format our
code that will make it agnostic towards device type and distributed setting using 
[Pytorch Lightning](https://pytorch-lightning.readthedocs.io/en/latest/).


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
[Distributed data parallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel) would require more than just wrapping your model in the appropriate class. It is in these situations that frameworks such as *Pytorch Lightning* really shines as the comes into play. As long as we format our model to the required
format of the framework we can enable distributed training with a single change of code.

### Exercises

1. Take a look at the `distributed_example.py` and `distributed_example.sh` files and try to understand
   them. They are essentially what you would need to implement yourself to get this working. Try to
   answer the following questions (HINT: try to google around a ):
   
   1. What is the function of the `DDP` wrapper?

   2. What is the function of the `DistributedSampler`?

   3. Why is it necessary to call `dist.barrier()` before passing a batch into the model?

   4. What does the different enviroment variables does in the `.sh` file

2. The last exercise have hopefully convinced you that it can be quite the trouble writing distributed training applications yourself.
   Luckly for us, `Pytorch-lightning` can take care of this for us such that we do not have to care about the specific details. To
   get your model training on multiple GPUs you need to change two arguments in the trainer: the `accelerator` flag and the `gpus` flag.
   In addition to this, you can read through this [guide](https://pytorch-lightning.readthedocs.io/en/latest/advanced/multi_gpu.html)
   about any additional steps you may need to do (for many of you, it should just work). Try running your model on multiple GPUs.

3. Try benchmarking your training using 1 and 2 gpus e.g. try running a couple of epochs and measure how long time it takes. 
   How much of a speedup can you actually get? Why can you not get a speedup of 2?

3. (Optional) Calling `self.log` by default will only log the result from process 1. Try chaning the `sync_dist` flag to accumulate
   the values across devices.
