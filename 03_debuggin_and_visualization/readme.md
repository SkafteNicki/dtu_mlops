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
of the exercises we will try to look at the [wandb](https://wandb.ai/site) logger. The great benefit of using wandb
over tensorboard is that it was build with colllaboration in mind (whereas tensorboard somewhat got it along the
way).

1. Start by creating an account at [wandb](https://wandb.ai/site). I recommend using your github account but feel
   free to choose what you want. When you are logged in you should get an API key of length 40. Copy this for later
   use (HINT: if you forgot to copy the API key, you can find it under settings).

2. Next install wandb on your laptop
   ```
   pip install wandb
   ```

3. Now connect to your wandb account
   ```
   wandb login
   ```
   you will be asked to provide the 40 length API key. The connection will be closed to the wandb server whenever
   you close the terminal. If using `wandb` in a notebook you need to manually close the connection using
   `wandb.finish()`
   ```

4. With it all setup we are now ready to incorporate `wandb` into our code. The interface is fairly simple, and
   this [guide](https://docs.wandb.ai/guides/integrations/pytorch) should give enough hints to get you through
   the exercise. (HINT: the two methods you need to call are `wandb.init` and `wandb.log`). To start with, logging
   the training loss of your model will be enough.

5. After running your model, checkout the webpage. Hopefully you should be able to see atleast 

6. Finally, lets create a report that you can share. Click the **Create report** botton where you choose the *blank*
   option. Then choose to include everything in the report 

7. To make sure that you have completed todays exercises, make the report shareable by clicking the *Share* botton
   and create *view-only-link*. Send the link to my email `nsde@dtu.dk`, so I can checkout your awesome work.

8. Feel free to experiment more with `wandb` as it is a great tool for logging, organising and sharing experiments.






