
### Profilers

Using profilers can help you find bottlenecks in your code. Luckly, PyTorch already comes with a build-in profiler.
Note that these exercises requires that you have pytorch v1.8.1 installed. You can always check which version you
currently have installed by writing (in python):

```python
import torch
print(torch.__version__)
```

https://pytorch.org/blog/introducing-pytorch-profiler-the-new-and-improved-performance-tool/

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


