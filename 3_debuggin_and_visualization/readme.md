
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

For these exercises we will initially be looking at incorporating [tensorboard](https://www.tensorflow.org/tensorboard) into our code, as it comes with native
support in pytorch

While tensorboard is a great logger for many things, more advanced loggers may be more suitable. For the remaining of the exercises we will try to look at the 
[wandb](https://wandb.ai/site) logger. 
