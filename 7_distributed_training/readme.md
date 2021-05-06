# 7. Distributed training

Today is about distributed training. We will start simple by making use of `DistributedData` which
is kind of the old standard for doing distributed training, and from there move on to format our
code that will make it agnostic towards device type and distributed setting using 
[Pytorch Lightning](https://pytorch-lightning.readthedocs.io/en/latest/)

## Distributed data loading

One common bottleneck in training deep learning models is the data loading (you maybe already saw this
during the profiling exercises). The reason

### Exercises

This exercise is intended to be done on the [labeled faces in the wild (LFW)](http://vis-www.cs.umass.edu/lfw/)
dataset. The dataset consist images of famous people extracted from the internet. The dataset had been used
to drive the field of facial verification, which you can read more about 
[here](https://machinelearningmastery.com/introduction-to-deep-learning-for-face-recognition/). We are going
imagine that this dataset cannot fit in memory, and your job is therefore to construct a data pipeline that
can be parallized based on loading the raw datafiles (.jpg) at runtime.

1. Download the dataset and extract to a folder. It does not matter if you choose the non-aligned or
   aligned version of the dataset.

2. We provide the `lfw_dataset.py` file where we have started the process of defining a data class. 
   Fill out the `__init__`, `__len__` and `__getitem__`. Note that `__getitem__` expect that you
   return a tuple `(img, label)` where `img` should be a [PIL Image](https://pillow.readthedocs.io/en/stable/)
   and `label` should be a unique integer indicating the identity of the person. We want the `img`
   to be `PIL` image so we can take advantage of [torchvision](https://pytorch.org/vision/stable/transforms.html)
   for data augmentation.  

2. Lets experiment with how much the number of workers have 

3. (Optional, requires access to GPU) If your dataset fits in GPU memory it is beneficial to set the
   `pin_memory` flag to `True`. By setting this flag we are essentially telling pytorch that they can
   lock the data in-place in memory which will make the transfer between the *host* (CPU) and the
   *device* (GPU) faster.

## Distributed Data

### Exercises


