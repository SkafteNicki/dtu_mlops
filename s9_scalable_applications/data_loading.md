
![Logo](../figures/icons/pytorch.png){ align=right width="130"}

# Distributed Data Loading

---

!!! note "Core Module"

One way that deep learning fundamentally changed the way we think about data in machine learning is that *more data is
always better*. This was very much not the case with more traditional machine learning algorithms (random forest,
support vector machines etc.) where a pleatau in performance was often reached for a certain amount of data and did not
improve if more was added. However, as deep learning models have become deeper and deeper and thereby more and more
data hungry performance seems to be ever increasing or at least not reaching a pleatau in the same way as for
traditional machine learning.

<figure markdown>
![Image](../figures/ml_data.png){ width="500" }
<figcaption> <a href="https://www.codesofinterest.com/2017/04/how-deep-should-deep-learning-be.html"> Image credit </a> </figcaption>
</figure>

As we are trying to feed more and more data into our models and obvious first question to ask is how to do this in a
efficient way. As an general rule of thumb we want the performance bottleneck to be the forward/backward e.g. the
actual computation in our neural network and not the data loading. By bottleneck we here refer to the part of our
pipeline that is restricting how fast we can process data. If data loading is our bottleneck, then our compute device
can sit idle while waiting for data to arrive, which is both inefficient and costly. For example if you are using a
cloud provider for training deep learning models, you are paying by the hour per device, and thus not using them fully
can be costly in the long run.

In the first set of exercises we are therefore going to focus on distributed data loading i.e. how do load data in
parallel to make sure that we always have data ready for our compute devices. We are in the following going to look
at what is going on behind the scene when we use Pytorch to parallelize data loading.

## A closer look on Data loading

Before we talk distributed applications it is important to understand the physical  layout of a standard CPU (the
brain of your computer).

<figure markdown>
![Image](../figures/cpu_layout.PNG){ width="500" }
</figure>

Most modern CPUs is a single chip that consist of multiple *cores*. Each core can further be divided into *threads*.
In most laptops the core count is 4 and commonly 2 threads per code. This means that the common laptop have 8 threads.
The number of threads a compute unit has is important, because that directly corresponds to the number of parallel
operations that can be executed i.e. one per thread. In a Python terminal you should be able to get the number of
cores in your machine by writing (try it):

```python
import multiprocessing
cores = multiprocessing.cpu_count()
print(f"Number of cores: {cores}, Number of threads: {2*cores}")
```

A distributed application is in general any kind of application that parallelizes some or all of it workload. We are
in these exercises only focusing on distributed data loading, which happens primarily only on the CPU. In `Pytorch` it
is easy to parallelize data loading if you are using their dataset/dataloader interface:

```python
from torch.utils.data import Dataset, DataLoader
class MyDataset(Dataset):
    def __init__(self, ...):
        # whatever logic is needed to init the data set
        self.data = ...

    def __getitem__(self, idx):
        # return one item
        return self.data[idx]

dataset = MyDataset()
dataloader = Dataloader(
    dataset,
    batch_size=8,
    num_workers=4  # this is the number of threds we want to parallize workload over
)
```

Lets take a deep dive into what happens when we request a batch from our dataloader e.g. ``next(dataloader)``. First we
must understand that we have a thread that plays the role of the *main* and the remaining threads (in the above example
we request 4) are called *workers*. When the dataloader is created, we create this structure and make sure that all
threads have a copy of our dataset definition so each can call the `__getitem__` method.

<figure markdown>
![Image](../figures/cpu_data_loading1.PNG){ width="500" }
</figure>

Then comes the actual part where we request a batch for data. Assume that we have a batch size of 8 and we do not do
any shuffeling. In this step the master thread then distributes the list of requested data points (`[0,1,2,3,4,5,6,7]`)
to the four worker threads. With 8 indices and 4 workers, each worker will receive 2 indices.

<figure markdown>
![Image](../figures/cpu_data_loading2.PNG){ width="500" }
</figure>

Each worker thread then calls `__getitem__` method for all the indices it has received. When all processes are done, the
loaded images datapoints gets send back to the master thread collected into a single structure/tensor.

<figure markdown>
![Image](../figures/cpu_data_loading3.PNG){ width="500" }
</figure>

Each arrow is corresponds to a communication between two threads, which is not a free operations. In total to get a
single batch (not counting the initial startup cost) in this example we need to do 8 communication operations. This may
seem like a small price to pay, but that may not be the case. If the process time of ``__getitem__`` is very low (data
is stored in memory, we just need to index to get it) then it does not make sense to use multiprocessing. The
computationally saving by doing the look-up operations in parallel is smaller than the communication cost there is
between the main thread and the workers. Multiprocessing makes sense when the process time of ``__getitem__`` is high
(data is probably stored on the harddrive).

It is this trade-off that we are going to investigate in the exercises.

### Exercises

This exercise is intended to be done on the
[labeled faces in the wild (LFW)](http://vis-www.cs.umass.edu/lfw/) dataset. The dataset consist images of famous people
extracted from the internet. The dataset had been used to drive the field of facial verification, which you can read
more about [here](https://viso.ai/deep-learning/deep-face-recognition/). We are going imagine that this dataset cannot
fit in memory, and your job is therefore to construct a data pipeline that can be parallelized based on loading the raw
datafiles (.jpg) at runtime.

1. Download the dataset and extract to a folder. It does not matter if you choose the non-aligned or aligned version of
    the dataset.

2. We provide the `lfw_dataset.py` file where we have started the process of defining a data class. Fill out the
    `__init__`, `__len__` and `__getitem__`. Note that `__getitem__` expect that you return a single `img` which should
    be a `torch.Tensor`. Loading should be done using [PIL Image](https://pillow.readthedocs.io/en/stable/), as `PIL`
    images is the default input format for [torchvision](https://pytorch.org/vision/stable/transforms.html) for
    transforms (for data augmentation).

3. Make sure that the script runs without any additional arguments

    ```bash
    python lfw_dataset.py
    ```

4. Visualize a single batch by filling out the codeblock after the first *TODO* right after defining the dataloader.
    The visualization should show when launching the script as

    ```bash
    python lfw_dataset.py -visualize_batch
    ```

    Hint: this [tutorial](https://pytorch.org/vision/stable/auto_examples/plot_visualization_utils.html#sphx-glr-auto-examples-plot-visualization-utils-py).

5. Experiment how the number of workers influences the performance. We have already provide code that will pass over 100
    batches from the dataset 5 times and calculate how long time it took, which you can play around with by calling

    ```bash
    python lfw_dataset.py -get_timing -num_workers 1
    ```

    Make a [errorbar plot](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.errorbar.html) with number of
    workers along the x-axis and the timing along the y-axis. The errorbars should correspond to the standard deviation
    over the 5 runs. HINT: if it is taking too long to evaluate, measure the time over less batches (set the
    `-batches_to_check` flag). Also if you are not seeing an improvement, try increasing the batch size (since data
    loading is parallelized per batch).

    For certain machines like the Mac with M1 chipset it is nessesary to set the `multiprocessing_context` flag in the
    dataloder to `"fork"`. This essentially tells the dataloader how the worker nodes should be created.

6. Retry the experiment where you change the data augmentation to be more complex:

    ```python
    lfw_trans = transforms.Compose([
        transforms.RandomAffine(5, (0.1, 0.1), (0.5, 2.0)),
        # add more transforms here
        transforms.ToTensor()
    ])
    ```

    by making the augmentation more computationally demanding, it should be easier to get an boost in performance when
    using multiple workers because the data augmentation is also executed in parallel.

7. (Optional, requires access to GPU) If your dataset fits in GPU memory it is beneficial to set the `pin_memory` flag
    to `True`. By setting this flag we are essentially telling Pytorch that they can lock the data in-place in memory
    which will make the transfer between the *host* (CPU) and the *device* (GPU) faster.

This ends the module on distributed data loading in Pytorch. If you want to go into more details we highly recommend
that you read [this paper](https://arxiv.org/pdf/2211.04908.pdf) that goes into great details on analyzing on how data
loading in Pytorch work and performance benchmarks.
![Logo](../figures/icons/lightning.png){ align=right width="130"}

# Distributed Training


---

In this module we are going to look at distributed training. Distributed training is one of the key ingredients
to all the awesome results that deep learning models are producing. For example:
[Alphafold](https://deepmind.com/blog/article/alphafold-a-solution-to-a-50-year-old-grand-challenge-in-biology)
the highly praised model from Deepmind that seems to have solved protein structure prediction, was trained
in a distributed fashion for a few weeks. The training was done on 16 TPUv3s (specialized hardware), which
is approximately equal to 100-200 modern GPUs. This means that training Alphafold without distributed training
on a single GPU (probably not even possible) would take a couple of years to train! Therefore, it is simply
impossible currently to train some of the state-of-the-art (SOTA) models within deep learning currently,
without taking advantage of distributed training.

When we talk about distributed training, there are a number of different paradigms that we may use to parallelize
our computations

* Data parallel (DP) training
* Distributed data parallel (DDP) training
* Sharded training

In this module we are going to look at data parallel training, which is the original way of doing parallel training and
distributed data parallel training which is an improved version of data parallel. If you want to know more about sharded
training which is the newest of the paradigms you can read more about it in this
[blog post](https://towardsdatascience.com/sharded-a-new-technique-to-double-the-size-of-pytorch-models-3af057466dba),
which describes how sharded can save over 60% of memory used during your training.

Finally, we want to note that for all the exercises in the module you are going to need a multi GPU setup. If you have
not already gained access to multi GPU machines on GCP (see the quotas exercises in
[this module](../s6_the_cloud/cloud_setup.md)) you will need to find another way of running the exercises. For
DTU Students I can recommend checking out [this optional module](../s10_extra/high_performance_clusters.md) on using the
high performance cluster (HPC) where you can get access to multi GPU resources.

## Data parallel

While data parallel today in general is seen as obsolete compared to distributed data parallel, we are still
going to investigate it a bit since it offers the most simple form of distributed computations in deep learning
pipeline.

In the figure below is shown both the *forward* and *backward* step in the data parallel paradigm

<figure markdown>
![Image](../figures/data_parallel.png){ width="1000" }
</figure>

The steps are the following:

* Whenever we try to do *forward* call e.g. `out=model(batch)` we take the batch and divide it equally between all
    devices. If we have a batch size of `N` and `M` devices each device will be sent `N/M` datapoints.

* Afterwards each device receives a copy of the `model` e.g. a copy of the weights that currently parametrizes our
    neural network.

* In this step we perform the actual *forward* pass in parallel. This is the actual steps that can help us scale
    our training.

* Finally we need to send back the output of each replicated model to the primary device.

Similar to the analysis we did of parallel data loading, we cannot always expect that this will actual take less time
than doing the forward call on a single GPU. If we are parallelizing over `M` devices, we essentially need to do `3xM`
communication calls to send batch, model and output between the devices. If the parallel forward call does not outweigh
this, then it will take longer.

In addition, we also have the *backward* path to focus on

* As the end of the *forward* collected the output on the primary device, this is also where the loss is accumulated.
    Thus, loss gradients are first calculated on the primary device

* Next we scatter the gradient to all the workers

* The workers then perform a parallel backward pass through their individual model

* Finally, we reduce (sum) the gradients from all the workers on the main process such that we can do gradient descend.

One of the big downsides of using data parallel is that all the replicas are destroyed after each *backward* call.
This means that we over and over again need to replicate our model and send it to the devices that are part of the
computations.

Even though it seems like a lot of logic is implementing data parallel into your code, in Pytorch we can very simply
enable data parallel training by wrapping our model in the
[nn.DataParallel](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html) class.

```python
from torch import nn
model = MyModelClass()
model = nn.DataParallel(model, device_ids=[0, 1])  # data parallel on gpu 0 and 1
preds = model(input)  # same as usual
```

### Exercises

Please note that the exercise only makes sense if you have access to multiple GPUs.

1. Create a new script (call it `data_parallel.py`) where you take a copy of model `FashionCNN`
    from the `fashion_mnist.py` script. Instantiate the model and wrap `torch.nn.DataParallel`
    around it such that it can be executed in data parallel.

2. Try to run inference in parallel on multiple devices (pass a batch multiple times and time it) e.g.

    ```python
    import time
    start = time.time()
    for _ in range(n_reps):
        out = model(batch)
    end = time.time()
    ```

    Does data parallel decrease the inference time? If no, can you explain why that may be? Try
    playing around with the batch size, and see if data parallel is more beneficial for larger batch sizes.

## Distributed data parallel

It should be clear that there is huge disadvantage of using the data parallel paradigm to scale your applications:
the model needs to replicated on each pass (because it is destroyed in the end), which requires a large transfer
of data. This is the main problem that distributed data parallel tries to solve.

<figure markdown>
![Image](../figures/distributed_data_parallel.png){ width="8000" }
</figure>

The two key difference between distributed data parallel and data parallel that we move the model update
(the gradient step) to happen on each device in parallel instead of only on the main device. This has the consequence
that we do not need to move replicate the model on each step, instead we just keep a local version on each device that
we keep updating. The full set of steps (as shown in the figure):

* Initialize an exact copy of the model on each device

* From disk (or memory) we start by loading data into a section of page-locked host memory per device. Page-locked
    memory is essentially a way to reverse a piece of a computers memory for a specific transfer that is going to
    happen over and over again to speed it up. The page-locked regions are loaded with non-overlapping data.

* Transfer data from page-locked memory to each device in parallel

* Perform *forward*  pass in parallel

* Do a all-reduce operation on the gradients. An all-reduce operation is a so call *all-to-all*  operation meaning
    that all processes send their own gradient to all other processes and also received from all other processes.

* Reduce the combined gradient signal from all processes and update the individual model in parallel. Since all
    processes received the same gradient information, all models will still be in sync.

Thus, in distributed data parallel we here end up only doing a single communication call between all processes, compared
to all the communication going on in data parallel. While all-reduce is a more expensive operation that many of the
other communication operations that we can do, because we only have to do a single we gain a huge performance boost.
Empirically distributed data parallel tends to be 2-3 times faster than data parallel.

However, this performance increase does not come for free. Where we could implement data parallel in a single line in
Pytorch, distributed data parallel is much more involving.

### Exercises

1. We have provided an example of how to do distributed data parallel training in Pytorch in the two
    files `distributed_example.py` and `distributed_example.sh`. You objective is to get a understanding of the necessary
    components in the script to get this kind of distributed training to work. Try to answer the following questions
    (HINT: try to Google around):

    1. What is the function of the `DDP` wrapper?

    2. What is the function of the `DistributedSampler`?

    3. Why is it necessary to call `dist.barrier()` before passing a batch into the model?

    4. What does the different environment variables do in the `.sh` file

2. Try to benchmark the runs using 1 and 2 GPUs

3. The first exercise have hopefully convinced you that it can be quite the trouble writing distributed training
    applications yourself. Luckily for us, `Pytorch-lightning` can take care of this for us such that we do not have to
    care about the specific details. To get your model training on multiple GPUs you need to change two arguments in the
    trainer: the `accelerator` flag and the `gpus` flag. In addition to this, you can read through this
    [guide](https://pytorch-lightning.readthedocs.io/en/latest/accelerators/gpu.html) about any additional steps you may
    need to do (for many of you, it should just work). Try running your model on multiple GPUs.

4. Try benchmarking your training using 1 and 2 gpus e.g. try running a couple of epochs and measure how long time it
    takes. How much of a speedup can you actually get? Why can you not get a speedup of 2?
