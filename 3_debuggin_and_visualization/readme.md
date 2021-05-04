# 3. Debugging, profiling and visualizing code

## Debugging

Debugging is very hard to teach and is one of the skills that just comes with experience. That said, you should
familar yourself with the build-in [python debugger](https://docs.python.org/3/library/pdb.html) as it may come in
handy during the course. 

### Exercises

We here provide a script `mnist_vae_bugs.py` which contains a number of bugs to get it running. Start by going over
the script and try to understand what is going on. Hereafter, try to get it running by solving the bugs. The following 
bugs exist in the script:

* One device bug (will only show if running on gpu, but try to find it anyways)
* One shape bug 
* One math bug 
* One training bug

Some of the bugs prevents the script from even running, while some of them influences the training dynamics.
Try to find them all. We also provide a working version called `vae_mnist_working.py` (but please try to find
the bugs before looking at the script). Succesfully debugging and running the script should produce three files: 
`orig_data.png`, `reconstructions.png`, `generated_samples.png`. 

## Profilers

Using profilers can help you find bottlenecks in your code. In this exercise we will look at two different
profilers, with the first one being the [cProfile](https://docs.python.org/3/library/profile.html), pythons
build in profiler.

### Exercises

1. Run the `cProfile` on the `vae_mnist_working.py` script. Hint: you can directly call the profiler on a
   script using the `-m` arg
   `python -m cProfile [-o output_file] [-s sort_order] (-m module | myscript.py) `

In addition to using pythons build-in profiler we will also investigate the profiler that is build into PyTorch already.
Note that these exercises requires that you have pytorch v1.8.1 installed. You can always check which version you
currently have installed by writing (in python):

```python
import torch
print(torch.__version__)
```

Also it will require us to have the tensorboard profiler plugin installed:

``` 
pip install torch_tb_profiler
```

For more info on the profiler see this [blogpost](https://pytorch.org/blog/introducing-pytorch-profiler-the-new-and-improved-performance-tool/)
and the [documentation](https://pytorch.org/docs/stable/profiler.html).

For this exercise

1. Start by going over the following [tutorial](https://pytorch.org/tutorials/beginner/profiler.html)

2. Secondly try to profile the `vae_mnist_working.py` script from the debugging exercises. Can you improve
something in the code?

3. Apply the profiler to your own code. 

### Experiement visualizers

While logging loss values to terminal, or plotting training curves in matplotlib may be enough doing smaller experiment,
there is no way around using a proper experiment tracker and visualizer when doing large scale experiments.

For these exercises we will initially be looking at incorporating [tensorboard](https://www.tensorflow.org/tensorboard) into our code, 
as it comes with native support in pytorch

1. Install tensorboard (does not require you to install tensorflow)
   ```pip install tensorboard```

2. Take a look at this [tutorial](https://pytorch.org/docs/stable/tensorboard.html)

3. Implement the summarywriter in your training script from the last session. The summarywrite should log both
   a scalar (`writer.add_scalar`) (atleast one) and a histogram (`writer.add_histogram`). Additionally, try log
   the computational graph (`writer.add_graph`).
   
4. Start tensorboard in a terminal
   ```tensorboard --logdir this/is/the/dir/tensorboard/logged/to```
   
5. Inspect what was logged in tensorboard

Experiement visualizers are especially useful for comparing values across training runs. Multiple runs often
stems from playing around with the hyperparameters of your model.

6. In your training script make sure the hyperparameters are saved to tensorboard (`writer.add_hparams`)

7. Run atleast two models with different hyperparameters, open them both at the same time in tensorboard
   Hint: to open multiple experiments in the same tensorboard they either have to share a root folder e.g.
   `experiments/experiment_1` and `experiments/experiment_2` you can start tensorboard as
   ```tensorboard --logdir experiments```
   or as
   ```tensorboard --logdir=name1:/path/to/logs/1,name2:/path/to/logs/2```

While tensorboard is a great logger for many things, more advanced loggers may be more suitable. For the remaining 
of the exercises we will try to look at the [wandb](https://wandb.ai/site) logger. 


