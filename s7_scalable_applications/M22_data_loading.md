---
layout: default
title: M22 - Distributed Data Loading
parent: S7 - Scalable applications
nav_order: 1
---

<img style="float: right;" src="../figures/icons/pytorch.png" width="130"> 

# Distributed Data Loading
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

{: .important }
> Core module

One way that deep learning fundamentally changed the way we think about data in machine learning is that *more data is
always better*. This was very much not the case with more traditional machine learning algorithms (random forest, 
support vector machines etc.) where a pleatau in performance was often reached for a certain amount of data and did not
improve if more was added. However, as deep learning models have become deeper and deeper and thereby more and more 
data hungry performance seems to be ever increasing or atl east not reaching a pleatau in the same way as for 
traditional machine learning.

<p align="center">
  <img src="../figures/ml_data.PNG" width="500">
  <br>
  <a href="https://www.codesofinterest.com/2017/04/how-deep-should-deep-learning-be.html"> Image credit </a>
</p>

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

### A closer look on Data loading 

Before we talk distributed applications it is important to understand the physical  layout of a standard CPU (the 
brain of your computer).

<p align="center">
    <img src="../figures/cpu_layout.PNG" width="500,">
</p>

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

<p align="center">
    <img src="../figures/cpu_data_loading1.PNG" width="500">
</p>

Then comes the actual part where we request a batch for data. Assume that we have a batch size of 8 and we do not do 
any shuffeling. In this step the master thread then distributes the list of requested data points (`[0,1,2,3,4,5,6,7]`) 
to the four worker threads. With 8 indices and 4 workers, each worker will receive 2 indices.

<p align="center">
    <img src="../figures/cpu_data_loading2.PNG" width="500">
</p>

Each worker thread then calls `__getitem__` method for all the indices it has received. When all processes are done, the 
loaded images datapoints gets send back to the master thread collected into a single structure/tensor.

<p align="center">
    <img src="../figures/cpu_data_loading3.PNG" width="500">
</p>

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
more about [here](https://machinelearningmastery.com/introduction-to-deep-learning-for-face-recognition/).
We are going imagine that this dataset cannot fit in memory, and your job is therefore to construct a data pipeline that 
can be parallelized based on loading the raw datafiles (.jpg) at runtime.

1. Download the dataset and extract to a folder. It does not matter if you choose the non-aligned or aligned version of 
   the dataset.

2. We provide the `lfw_dataset.py` file where we have started the process of defining a data class. Fill out the 
   `__init__`, `__len__` and `__getitem__`. Note that `__getitem__` expect that you return a single `img` which should 
   be a `torch.Tensor`. Loading should be done using [PIL Image](https://pillow.readthedocs.io/en/stable/), as `PIL` 
   images is the default input format for [torchvision](https://pytorch.org/vision/stable/transforms.html) for 
   transforms (for data augmentation).  

3. Make sure that the script runs without any additional arguments
   ```
   python lfw_dataset.py
   ```

4. Visualize a single batch by filling out the codeblock after the first *TODO* right after defining the dataloader. 
   The visualization should show when launching the script as
   ```
   python lfw_dataset.py -visualize_batch
   ```
   Hint: this [tutorial](https://pytorch.org/vision/stable/auto_examples/plot_visualization_utils.html#sphx-glr-auto-examples-plot-visualization-utils-py).

5. Experiment how the number of workers influences the performance. We have already provide code that will pass over 100 
   batches from the dataset 5 times and calculate how long time it took, which you can play around with by calling
   ```
   python lfw_dataset.py -get_timing -num_workers 1
   ```
   Make a [errorbar plot](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.errorbar.html) with number of 
   workers along the x-axis and the timing along the y-axis. The errorbars should correspond to the standard deviation 
   over the 5 runs. HINT: if it is taking too long to evaluate, measure the time over less batches (set the 
   `-batches_to_check` flag). Also if you are not seeing an improvement, try increasing the batch size (since data 
   loading is parallelized per batch).

6. (Optional, requires access to GPU) If your dataset fits in GPU memory it is beneficial to set the `pin_memory` flag 
   to `True`. By setting this flag we are essentially telling Pytorch that they can lock the data in-place in memory 
   which will make the transfer between the *host* (CPU) and the *device* (GPU) faster.

This ends the module on distributed data loading in Pytorch. If you want to go into more details we highly recommend
that you read [this paper](https://arxiv.org/pdf/2211.04908.pdf) that goes into great details on analyzing on how data
loading in Pytorch work and performance benchmarks.
